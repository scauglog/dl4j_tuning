package aperture.tuning.predictions.model

import org.deeplearning4j.nn.graph.ComputationGraph

object ModelLoader {
  def initializeNewModel(features: Seq[String], networkConfiguration: NetworkConfiguration): ComputationGraph = {
    val modelConfig = ModelConfigurationBuilder.buildConf(features.length, networkConfiguration)
    val network: ComputationGraph = new ComputationGraph(modelConfig)
    network.init()
    network
  }
}
