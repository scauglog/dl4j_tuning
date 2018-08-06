package aperture.tuning.predictions.tuning

import aperture.tuning.predictions.model.{InMemoryMultiDataSetIterator, ModelTraining}

case class EvaluationConfiguration(networkConfiguration: String,
                                   maeTest: Double,
                                   rmseTest: Double,
                                   maeTrain: Double,
                                   rmseTrain: Double) {
  override def toString: String = {
    s"""networkConfiguration: $networkConfiguration
       |maeTest: $maeTest
       |rmseTest: $rmseTest
       |maeTrain: $maeTrain
       |rmseTrain: $rmseTrain
    |""".stripMargin
  }
}

object EvaluateModel {
  def evaluateModel(trainingModel: ModelTraining, dataSetIt: InMemoryMultiDataSetIterator)
  : (Double, Double)= {
    val evaluation = trainingModel.evaluateModel(dataSetIt)
    val mae = evaluation.meanAbsoluteError(0)
    val rmse = evaluation.rootMeanSquaredError(0)
    (mae, rmse)
  }
}
