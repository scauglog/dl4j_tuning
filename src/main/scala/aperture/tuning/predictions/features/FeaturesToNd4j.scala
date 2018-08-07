package aperture.tuning.predictions.features

import aperture.tuning.predictions.FeaturesAndTarget
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset
import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.factory.Nd4j

object FeaturesToNd4j {
  def transformFeaturesAndTargetToMultiDatasetShaped3(featuresAndTarget: Array[FeaturesAndTarget], timeSteps: Int)
  : MultiDataSet = {
    val numFeatures: Int = featuresAndTarget.head.features.length / timeSteps
    val features: Array[Double] = featuresAndTarget.flatMap(_.features)
    val featuresArray: INDArray = Nd4j.create(features, Array(featuresAndTarget.length, numFeatures, timeSteps))
    val bidsArray: INDArray = Nd4j.create(featuresAndTarget.map(_.bid), Array(featuresAndTarget.length, 1))
    val labelsArray: Array[INDArray] = Array(Nd4j.create(featuresAndTarget.map(_.target), Array(featuresAndTarget.length,1)))
    val inputsArray: Array[INDArray] = Array(featuresArray, bidsArray)
    new dataset.MultiDataSet(inputsArray, labelsArray)
  }
}
