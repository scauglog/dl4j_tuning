package aperture.tuning.predictions.model

import com.typesafe.scalalogging.Logger
import org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.{MaxEpochsTerminationCondition, ScoreImprovementEpochTerminationCondition}
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer
import org.deeplearning4j.eval.RegressionEvaluation
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.parallelism.EarlyStoppingParallelTrainer
import org.deeplearning4j.ui.stats.StatsListener
import org.nd4j.jita.conf.CudaEnvironment
import org.nd4j.linalg.jcublas.JCublasBackend
import org.deeplearning4j.optimize.listeners.PerformanceListener
import org.nd4j.linalg.dataset
import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex.interval
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.EarlyStoppingResult
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener

object ModelTraining {
  val DEFAULT_NUMBER_OF_EPOCHS: Int = 50
}

class ModelTraining(network: ComputationGraph, batchSize: Int, numberOfEpochs: Int, feedUI: Boolean = false) {
  val logger: Logger = Logger(getClass)
  val averagingFrequency: Int = 50
  val prefetchBufferMultiplier = 16
  val evaluateEveryNEpoch = 5
  val trainSetPercent = 0.8
  val gcPauseMilliSec = 60000
  val maxEpochWithoutImprovement = 5
  val minScoreChange = 5e-5

  private val UIUrl = "http://127.0.0.1:9000"

  def train(featuresAndTarget: MultiDataSet): ComputationGraph =  {
    if (feedUI) initializeUIConnection()

    increaseGCTime()

    val (trainDatasetIterator, evalDatasetIterator) = initializeDatasetIterator(featuresAndTarget, batchSize)
    trainWithIterator(trainDatasetIterator, evalDatasetIterator)
  }

  def trainWithIterator(train: InMemoryMultiDataSetIterator, eval: InMemoryMultiDataSetIterator): ComputationGraph = {
    val saver = new InMemoryModelSaver[ComputationGraph]()
    val earlyStopConf = new EarlyStoppingConfiguration.Builder[ComputationGraph]()
      .epochTerminationConditions(new MaxEpochsTerminationCondition(numberOfEpochs), new ScoreImprovementEpochTerminationCondition(maxEpochWithoutImprovement, minScoreChange))
      .scoreCalculator(new DataSetLossCalculator(eval, true))
      .evaluateEveryNEpochs(evaluateEveryNEpoch)
      .modelSaver(saver)
      .build


    val listener = new LoggingEarlyStoppingListener()
    val trainer = if (isGPUBackend) {
      increaseCudaMemory()
      val workers = Nd4j.getAffinityManager.getNumberOfDevices
      val prefetchBuffer = prefetchBufferMultiplier * Nd4j.getAffinityManager.getNumberOfDevices
      new EarlyStoppingParallelTrainer[ComputationGraph](earlyStopConf, network, null, train, listener, workers, prefetchBuffer, averagingFrequency)
    } else {
      new EarlyStoppingGraphTrainer(earlyStopConf, network, train, listener)
    }

    val result = trainer.fit()
    logger.info(result.getTerminationDetails)
    logger.info(s"stop at epoch: ${result.getBestModelEpoch}, score: ${result.getBestModelScore}")

    result.getBestModel
  }

  private def initializeDatasetIterator(featuresAndTarget: MultiDataSet, batchSize: Int)
  : (InMemoryMultiDataSetIterator, InMemoryMultiDataSetIterator) = {
    featuresAndTarget.shuffle()
    val (train, eval) = splitDataset(featuresAndTarget, trainSetPercent)
    (new InMemoryMultiDataSetIterator(train, batchSize),new InMemoryMultiDataSetIterator(eval, batchSize))
  }

  private def initializeUIConnection(): Unit = {
    val remoteUIRouter = new RemoteUIStatsStorageRouter(UIUrl)
    network.setListeners(Seq(new StatsListener(remoteUIRouter), new PerformanceListener(32)): _*)
  }

  def evaluateModel(datasetIterator: InMemoryMultiDataSetIterator): RegressionEvaluation = {
    network.evaluateRegression(datasetIterator)
  }

  def isGPUBackend: Boolean = Nd4j.getBackend.isInstanceOf[JCublasBackend]

  def increaseGCTime(): Unit = Nd4j.getMemoryManager.setAutoGcWindow(gcPauseMilliSec)

  def increaseCudaMemory(): Unit = {
    CudaEnvironment.getInstance().getConfiguration
      .allowMultiGPU(true)
      // cross-device access is used for faster model averaging over pcie
      .allowCrossDeviceAccess(true)
  }

  def splitDataset(multiDataset: MultiDataSet, percent: Double): (MultiDataSet, MultiDataSet) = {
    val numExample = multiDataset.getFeatures(0).size(0)
    val split = (numExample * percent).toInt
    val trainInputs = multiDataset.getFeatures.map(feat => feat.get(interval(0, split)))
    val trainLabels = multiDataset.getLabels.map(label => label.get(interval(0, split)))
    val train = new dataset.MultiDataSet(trainInputs, trainLabels).asInstanceOf[MultiDataSet]

    val evalInputs = multiDataset.getFeatures.map(feat => feat.get(interval(split, numExample)))
    val evalLabels = multiDataset.getLabels.map(label => label.get(interval(split, numExample)))
    val eval = new dataset.MultiDataSet(evalInputs, evalLabels).asInstanceOf[MultiDataSet]

    (train, eval)
  }
}

class LoggingEarlyStoppingListener extends EarlyStoppingListener[ComputationGraph] {
  private val log: Logger = Logger(getClass)

  override def onStart(esConfig: EarlyStoppingConfiguration[ComputationGraph], net: ComputationGraph): Unit = {
    log.info("EarlyStopping: onStart called")
  }

  override def onEpoch(epochNum: Int, score: Double, esConfig: EarlyStoppingConfiguration[ComputationGraph], net: ComputationGraph): Unit = {
    log.info(s"EarlyStopping: onEpoch called (epochNum=$epochNum, score=$score)")
  }

  override def onCompletion(esResult: EarlyStoppingResult[ComputationGraph]): Unit = {
    log.info(s"EarlyStopping: onCompletion called (result: $esResult)")
  }
}