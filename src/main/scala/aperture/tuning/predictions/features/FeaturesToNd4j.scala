package aperture.tuning.predictions.features

import aperture.tuning.predictions.FeaturesAndTarget
import org.apache.spark.sql.Dataset
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset
import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.factory.Nd4j

object FeaturesToNd4j {
  def transformFeaturesAndTargetToMultiDatasetShaped3(featuresAndTarget: Dataset[FeaturesAndTarget], timeSteps: Int)
  : MultiDataSet = {
    val featuresAndTargetArray = featuresAndTarget.collect
    val numFeatures: Int = featuresAndTargetArray.head.features.length / timeSteps
    val features: Array[Double] = featuresAndTargetArray.flatMap(_.features)
    val featuresArray: INDArray = Nd4j.create(features, Array(featuresAndTargetArray.length, numFeatures, timeSteps))
    val bidsArray: INDArray = Nd4j.create(featuresAndTargetArray.map(_.bid), Array(featuresAndTargetArray.length, 1))
    val labelsArray: Array[INDArray] = Array(Nd4j.create(featuresAndTargetArray.map(_.target), Array(featuresAndTargetArray.length,1)))
    val inputsArray: Array[INDArray] = Array(featuresArray, bidsArray)
    new dataset.MultiDataSet(inputsArray, labelsArray)
  }
}
