package aperture.tuning.predictions.tuning

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import aperture.tuning.predictions.features.FeaturesToNd4j
import aperture.tuning.predictions.{CsvRow, FeaturesAndTarget}
import com.typesafe.scalalogging.Logger
import org.deeplearning4j.nn.graph.ComputationGraph
import aperture.tuning.predictions.model._
import aperture.tuning.helpers.JobMetadata
import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.factory.Nd4j
import scala.io.Source

object TuningMain {
  val logger: Logger = Logger(getClass)

  val DEFAULT_CONFIG = Config(inputPath = "inputs",
    outputdirPath = "result",
    numberOfEpochs = ModelTraining.DEFAULT_NUMBER_OF_EPOCHS,
    evaluationConfiguration = "")

  val defaultTuningConfiguration: TuningConfiguration = Configurations.defaultTuningConfiguration

  def main(args: Array[String]): Unit = {
    val conf = initApp(args)
    val tuningDate = LocalDateTime.now.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME).replace(":", "-")
    val baseFeatures = Seq("feat1","feat2","feat3")
    val numberOfLags = 3
    var loopCount = 0

    for (dummy <- 0 to 1000) yield {
      val trainSet = readCsvFile(conf.inputPath + "/train/data_dup.csv")
      val testSet = readCsvFile(conf.inputPath + "/test/data.csv")

      val features = baseFeatures

      val preprocessedDataTrain = preprocessPerformanceData(features, numberOfLags)(trainSet)

      val preprocessedDataTest = preprocessPerformanceData(features, numberOfLags)(testSet)

      val trainIt = new InMemoryMultiDataSetIterator(preprocessedDataTrain, batchSize = 256)
      val evalIt = new InMemoryMultiDataSetIterator(preprocessedDataTest, batchSize = 256)

      Configurations.generateConfigurations(defaultTuningConfiguration).foreach { networkConfiguration =>
        val (model, confWithEvaluationConfiguration) =
          trainModel(networkConfiguration, trainIt, evalIt, conf, numberOfLags, features)
        saveModelAndMetadata(model, confWithEvaluationConfiguration, s"${dummy}_$loopCount", tuningDate)

        loopCount += 1
      }
      Nd4j.getWorkspaceManager.destroyAllWorkspacesForCurrentThread()
      Nd4j.getMemoryManager.invokeGc()
    }
  }

  def readCsvFile(path: String): Array[CsvRow] = {
    val bufferedSource = Source.fromFile(path)
    val data = bufferedSource.getLines().toArray.map(CsvRow.apply)
    bufferedSource.close
    data
  }

  def preprocessPerformanceData(features: Seq[String], numberOfLags: Int)(inputDF: Array[CsvRow]): MultiDataSet = {
    val preprocessedData = inputDF.map { row =>
      FeaturesAndTarget(
        bid = row.bid,
        target = row.target,
        features = Array(row.feat1, row.feat2, row.feat3, row.feat1_1, row.feat2_1, row.feat3_1, row.feat1_2, row.feat2_2, row.feat3_2)
      )
    }
    FeaturesToNd4j.transformFeaturesAndTargetToMultiDatasetShaped3(preprocessedData, numberOfLags)
  }

  def trainModel(networkConfiguration: NetworkConfiguration,
                 trainIt: InMemoryMultiDataSetIterator,
                 evalIt: InMemoryMultiDataSetIterator,
                 conf: Config, numberOfLags: Int, features: Seq[String]): (ComputationGraph, TuningMain.Config) = {
    val network = ModelLoader.initializeNewModel(features, networkConfiguration)

    val trainingModel = new ModelTraining(network = network, numberOfEpochs = conf.numberOfEpochs, batchSize = 256)
    val model = trainingModel.trainWithIterator(trainIt, evalIt)

    val (maeTest, rmseTest) = EvaluateModel.evaluateModel(trainingModel, trainIt)
    val (maeTrain, rmseTrain) = EvaluateModel.evaluateModel(trainingModel, evalIt)

    val confWithEvaluationConfiguration =
      conf.copy(evaluationConfiguration = EvaluationConfiguration(
        networkConfiguration.toString,
        maeTest, rmseTest,
        maeTrain, rmseTrain).toString,
        features = features.mkString(" | ")
      )

    (model, confWithEvaluationConfiguration)
  }

  def saveModelAndMetadata(model: ComputationGraph, conf: Config, loopCount: String, tuningDate: String): Unit = {

    val outputDirectory = ModelPath.modelDirectoryPath(conf.outputdirPath, tuningDate)

    val outputDirectoryWithNumberVersion = outputDirectory + loopCount
    logger.info("Output directory: " + outputDirectoryWithNumberVersion)

    JobMetadata.saveMetadataToYamlOnHdfs(
      conf,
      ModelPath.getConfigPath(outputDirectoryWithNumberVersion)
    )

    ModelSaver.writeModel(model, ModelPath.getModelZipPath(outputDirectoryWithNumberVersion))
  }

  private def initApp(args: Array[String]) = {
    val parser = new scopt.OptionParser[Config]("TuningMain") {
      head("TuningMain", "1.0")

      opt[String]('i', "input-path") action { (inputPath, previousConfig) =>
        previousConfig.copy(inputPath = inputPath)
      } text "path on HDFS for input"

      opt[Int]('e', "number-of-epochs") action { (numberOfEpochs, previousConfig) =>
        previousConfig.copy(numberOfEpochs = numberOfEpochs)
      } text "number of epochs"

      opt[String]('o', "output-dir-path") action { (outputDirPath, previousConfig) =>
        previousConfig.copy(outputdirPath = outputDirPath)
      } text "output path on HDFS"

    }
    parser.parse(args, DEFAULT_CONFIG).get
  }

  case class Config(inputPath: String,
                    outputdirPath: String,
                    numberOfEpochs: Int,
                    evaluationConfiguration: String,
                    features: String = "")
}
