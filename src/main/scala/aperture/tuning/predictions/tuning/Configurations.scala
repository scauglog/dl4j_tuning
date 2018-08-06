package aperture.tuning.predictions.tuning

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.RmsProp
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import aperture.tuning.predictions.model._

case class TuningConfiguration(numberOfLags: Seq[Int],
                               optimizers: Seq[OptimizationAlgorithm],
                               layersConfigurations: Seq[Seq[LayerConfiguration]],
                               lossFunction: Seq[LossFunction],
                               updaters: Seq[Updater],
                               weightInits: Seq[WeightInit],
                               addFeatures: Seq[Seq[String]])

object Configurations {
  val defaultTuningConfiguration = TuningConfiguration(
    numberOfLags = Seq(28),
    optimizers = Seq(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT),
    layersConfigurations = Seq(
      Seq(LayerConfiguration(activationFunction = Activation.RELU, numberOfNeuronsOut = 64),
        LayerConfiguration(activationFunction = Activation.RELU, numberOfNeuronsOut = 32),
        LayerConfiguration(activationFunction = Activation.RELU, numberOfNeuronsOut = 16),
        LayerConfiguration(activationFunction = Activation.RELU, numberOfNeuronsOut = 8)
      )
    ),
    lossFunction = Seq(
      LossFunction.MEAN_ABSOLUTE_ERROR
    ),
    updaters = Seq(
      RmsPropUpdater(new RmsProp(0.05))
    ),
    weightInits = Seq(
      WeightInit.XAVIER
    ),
    addFeatures = featuresCombination(Seq(
      "feat_add_1",
      "feat_add_2",
      "feat_add_3"
    ))
  )

  def featuresCombination(features: Seq[String]): Seq[Seq[String]] = {
    features.toSet[String].subsets.map(_.toSeq).toSeq ++ Seq()
  }

  def generateConfigurations(configuration: TuningConfiguration): Seq[NetworkConfiguration] =
    for (optimizer <- configuration.optimizers;
         layerConf <- configuration.layersConfigurations;
         lossFunction <- configuration.lossFunction;
         updater <- configuration.updaters;
         weightInit <- configuration.weightInits)
      yield NetworkConfiguration(optimizer, layerConf, lossFunction, updater, weightInit)
}
