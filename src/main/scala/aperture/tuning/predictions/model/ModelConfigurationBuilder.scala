package aperture.tuning.predictions.model

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.constraint.NonNegativeConstraint
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex
import org.deeplearning4j.nn.conf.layers.{DenseLayer, LSTM, LossLayer}
import org.deeplearning4j.nn.conf.{BackpropType, ComputationGraphConfiguration, NeuralNetConfiguration, WorkspaceMode}
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.{Adam, Nesterovs, RmsProp, Sgd}
import org.nd4j.linalg.lossfunctions.LossFunctions


sealed trait Updater

case class RmsPropUpdater(rmsProp: RmsProp) extends Updater

case class SgdUpdater(sgd: Sgd) extends Updater

case class MomentumUpdater(nesterovs: Nesterovs) extends Updater

case class AdamUpdater(adam: Adam) extends Updater

case class LayerConfiguration(activationFunction: Activation, numberOfNeuronsOut: Int)

case class NetworkConfiguration(optimizer: OptimizationAlgorithm,
                                layersConfiguration: Seq[LayerConfiguration],
                                lossFunction: LossFunctions.LossFunction,
                                updater: Updater,
                                weightInit: WeightInit) {
  override def toString: String = {
    s"""|optimizer: $optimizer
       |layersConfiguration:\n  ${
      layersConfiguration.zipWithIndex.map {
        case (layerConfiguration, index) =>
          s"layer number ${index + 1}: $layerConfiguration"
      }.mkString("\n  ")
    }
       |lossFunction: $lossFunction
       |updater: $updater
       |weightInit: $weightInit
       |""".stripMargin
  }
}

object ModelConfigurationBuilder {
  val lstmLayerSize: Int = 64

  def defaultNetworkConfiguration(learningRate: Double): NetworkConfiguration =
    NetworkConfiguration(optimizer = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT,
      layersConfiguration = Seq(
        LayerConfiguration(activationFunction = Activation.RELU, numberOfNeuronsOut = lstmLayerSize),
        LayerConfiguration(activationFunction = Activation.RELU, numberOfNeuronsOut = lstmLayerSize / 2),
        LayerConfiguration(activationFunction = Activation.RELU, numberOfNeuronsOut = lstmLayerSize / 4),
        LayerConfiguration(activationFunction = Activation.RELU, numberOfNeuronsOut = lstmLayerSize / 8)
      ),
      lossFunction = LossFunctions.LossFunction.MSE,
      updater = RmsPropUpdater(new RmsProp(learningRate)),
      weightInit = WeightInit.XAVIER
    )

  def buildConf(numberOfFeatures: Int, networkConfiguration: NetworkConfiguration = defaultNetworkConfiguration(0.01)): ComputationGraphConfiguration = {
    val neuralBuilder = new NeuralNetConfiguration.Builder()
      .trainingWorkspaceMode(WorkspaceMode.ENABLED)
      .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
      .optimizationAlgo(networkConfiguration.optimizer)
      .weightInit(networkConfiguration.weightInit)
      .biasInit(0.0)
      .updater(networkConfiguration.updater match {
        case rmsProp: RmsPropUpdater => rmsProp.rmsProp
        case sgd: SgdUpdater => sgd.sgd
        case momentum: MomentumUpdater => momentum.nesterovs
        case adam: AdamUpdater => adam.adam
      })
      .graphBuilder()
      .addInputs("input_lstm", "input_cpc")
      .addLayer("first_lstm_layer",
        new LSTM.Builder()
          .nIn(numberOfFeatures)
          .nOut(networkConfiguration.layersConfiguration.head.numberOfNeuronsOut)
          .activation(networkConfiguration.layersConfiguration.head.activationFunction)
          .build(),
        "input_lstm")
      .addVertex("lastTimeStep", new LastTimeStepVertex("input_lstm"), "first_lstm_layer")
      .addVertex("merge", new MergeVertex(),
        "lastTimeStep", "input_cpc")

        networkConfiguration.layersConfiguration.zipWithIndex.foreach {
          case (layerConf, index) if index == 0 =>
            neuralBuilder.addLayer("0_dense",
              new DenseLayer.Builder()
                .constrainWeights(new NonNegativeConstraint())
                .nIn(networkConfiguration.layersConfiguration.head.numberOfNeuronsOut + 1)
                .nOut(networkConfiguration.layersConfiguration(1).numberOfNeuronsOut)
                .activation(networkConfiguration.layersConfiguration(1).activationFunction)
                .build,
              "merge")
          case (layerConf, index) if index < networkConfiguration.layersConfiguration.length - 1 =>
            neuralBuilder.addLayer(s"${index}_dense",
              new DenseLayer.Builder()
                .constrainWeights(new NonNegativeConstraint())
                .nIn(networkConfiguration.layersConfiguration(index).numberOfNeuronsOut)
                .nOut(networkConfiguration.layersConfiguration(index + 1).numberOfNeuronsOut)
                .activation(networkConfiguration.layersConfiguration(index + 1).activationFunction)
                .build,
              s"${index - 1}_dense")

          case _ =>
        }

    val lastLayerIndex = networkConfiguration.layersConfiguration.length - 1
    neuralBuilder.addLayer("output_dense", new DenseLayer.Builder()
      .constrainWeights(new NonNegativeConstraint())
      .nIn(networkConfiguration.layersConfiguration(lastLayerIndex).numberOfNeuronsOut)
      .nOut(1)
      .activation(Activation.IDENTITY)
      .build(), s"${lastLayerIndex - 1}_dense")
      .addLayer("output_layer", new LossLayer.Builder(networkConfiguration.lossFunction)
        .activation(Activation.IDENTITY)
        .build(), "output_dense")
      .setOutputs("output_layer")
      .backpropType(BackpropType.Standard)
      .pretrain(false)
      .backprop(true)
      .build
  }
}
