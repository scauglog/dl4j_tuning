package aperture.tuning.helpers

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataOutputStream, FileSystem, Path}

object IO {
  def createHdfsOutputStream(path: String): FSDataOutputStream = {
    val hdfs = FileSystem.get(new Configuration())
    val dest = new Path(path)
    hdfs.create(dest)
  }

  def writeStringToHdfs(inputData: String, outputPath: String, hdfs: FileSystem): Unit = {
    val dest = new Path(outputPath)

    val output = hdfs.create(dest, true)
    try {
      output.writeBytes(inputData)
    } finally {
      output.close()
    }
  }

  def combinePaths(path1: String, path2: String): String = {
    s"$path1/$path2"
  }
}
