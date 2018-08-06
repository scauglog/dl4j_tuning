package aperture.tuning.predictions.model

import java.io.File

import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.util.ModelSerializer
import aperture.tuning.helpers.{IO, PathProtocol}

object ModelSaver {
  def writeModel(model: ComputationGraph, outputPath: String): Unit = {
    if (PathProtocol.isHdfsPath(outputPath)) {
      val output = IO.createHdfsOutputStream(PathProtocol.hdfsPathWithoutProtocol(outputPath))
      ModelSerializer.writeModel(model, output, true)
    } else {
      val targetDirectory = new File(outputPath).getParentFile
      if (!targetDirectory.exists()) targetDirectory.mkdirs()
      ModelSerializer.writeModel(model, outputPath, true)
    }
  }
}
