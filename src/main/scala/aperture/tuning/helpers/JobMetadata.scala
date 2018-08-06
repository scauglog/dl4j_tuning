package aperture.tuning.helpers

import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.yaml.snakeyaml.{DumperOptions, Yaml}

import scala.collection.JavaConverters._

object JobMetadata {
  def saveMetadataToYamlOnHdfs(caseClass: AnyRef, outputPath: String,
                               extraParams: Map[String, Any] = Map.empty):
  Unit = {
    val parametersMap = extractFieldsFromCaseClass(caseClass)
    parametersMap.putAll(extraParams.asJava)
    val parametersYaml = initializeYaml.dump(parametersMap)
    IO.writeStringToHdfs(parametersYaml, outputPath, FileSystem.get(new Configuration))
  }

  private def extractFieldsFromCaseClass(caseClass: AnyRef): util.HashMap[String, Any] = {
    val fields = new util.HashMap[String, Any]()
    (fields /: caseClass.getClass.getDeclaredFields) { (fields, field) =>
      field.setAccessible(true)
      fields.put(field.getName, field.get(caseClass) match {
        case f: LocalDate => f.format(DateTimeFormatter.ISO_LOCAL_DATE)
        case f => f
      })
      fields
    }
  }

  private def initializeYaml: Yaml = new Yaml(initializeYamlOptions)

  private def initializeYamlOptions: DumperOptions = {
    val options = new DumperOptions
    options.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK)
    options
  }
}
