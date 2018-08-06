package aperture.tuning.predictions.model

import aperture.tuning.helpers.IO

object ModelPath {
  val MODEL_FILENAME = "model.zip"
  val CONFIG_FILENAME = "config.yaml"
  val SCALER_DIRECTORY = "scaler/"

  def currentModelDirectoryPath(dirPath: String, label: String): String =
    List(dirPath, label, "current").mkString("/").toString

  def modelDirectoryPath(dirPath: String, day: String): String =
    List(dirPath, day).mkString("/").toString + "/v"

  def getScalerPath(outputDirectory: String): String =
    IO.combinePaths(outputDirectory, SCALER_DIRECTORY)

  def getModelZipPath(outputDirectory: String): String =
    IO.combinePaths(outputDirectory, MODEL_FILENAME)

  def getConfigPath(outputDirectory: String): String =
    IO.combinePaths(outputDirectory, CONFIG_FILENAME)
}
