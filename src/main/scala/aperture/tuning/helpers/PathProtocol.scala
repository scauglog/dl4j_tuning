package aperture.tuning.helpers

object PathProtocol {
  def isHdfsPath(path: String): Boolean = path.startsWith("hdfs://")
  def hdfsPathWithoutProtocol(path: String): String = path.drop("hdfs://".length)
}
