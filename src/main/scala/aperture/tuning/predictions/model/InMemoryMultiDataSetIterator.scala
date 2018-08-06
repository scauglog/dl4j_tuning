package aperture.tuning.predictions.model

import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.dataset
import org.nd4j.linalg.indexing.NDArrayIndex.interval


class InMemoryMultiDataSetIterator(multiDataSet: MultiDataSet, batchSize: Int) extends MultiDataSetIterator {
  private var currentBatch = 0
  private val numExample = multiDataSet.getFeatures(0).size(0)
  private val totalBatches = numExample / batchSize

  private val inMemoryDataset = (0L to totalBatches).toArray.map { nextSlice }

  private def nextSlice(batchNum: Long): MultiDataSet = {
    val start = batchNum * batchSize
    val end = if (batchNum == totalBatches) {
      numExample
    } else {
      (batchNum + 1) * batchSize
    }
    val inputs = multiDataSet.getFeatures.map(feat => feat.get(interval(start, end)))
    val labels = multiDataSet.getLabels.map(label => label.get(interval(start, end)))
    new dataset.MultiDataSet(inputs, labels).asInstanceOf[MultiDataSet]
  }

  override def next(sampleSize: Int): MultiDataSet = {
    if (sampleSize != batchSize)
      throw new IllegalArgumentException("A preloaded dataset iterator cannot supply datasets of varying sizes.")
    else
      next
  }

  override def reset(): Unit = {
    currentBatch = 0
  }

  override def resetSupported = true

  override def asyncSupported = true

  override def hasNext: Boolean = currentBatch < totalBatches

  override def next: MultiDataSet = {
    currentBatch += 1
    inMemoryDataset(currentBatch-1)
  }

  override def getPreProcessor: MultiDataSetPreProcessor = {
    throw new UnsupportedOperationException("Not supported")
  }

  override def setPreProcessor(preProcessor: MultiDataSetPreProcessor): Unit = {
    throw new UnsupportedOperationException("Not supported")
  }
}