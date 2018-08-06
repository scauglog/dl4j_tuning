package aperture.tuning.predictions.evaluation

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{abs, col, count, max, min, sum, pow}
import org.apache.spark.sql.types.DoubleType

case class EvaluationMetrics(mae: Double, rmse: Double, min: Double, max: Double, keywordPredicted: Long) {
  def print(): Unit = {
    println("finalDf: keyword count", this.keywordPredicted)
    println("MAE", this.mae)
    println("RMSE", this.rmse)
    println("min", this.min)
    println("max", this.max)
  }

}

object EvaluationMetrics {
  def calculate(df: DataFrame, label: String): EvaluationMetrics = {
    val mae = "mae"
    val rmse = "rmse"
    val minCol = "min"
    val maxCol = "max"
    val predicted = "predicted"
    val diff = "diff"
    val cpt = "cpt"

    val metrics = df.select(
      col(getPredictedVariableName(label)).cast(DoubleType).as(predicted),
      (col(getPredictedVariableName(label)).cast(DoubleType) - col(label).cast(DoubleType)).as(diff)
    ).agg(max(col(predicted)).as(maxCol),
        min(col(predicted)).as(minCol),
        count(col(predicted)).as(cpt),
        sum(abs(col(diff))).as(mae),
        sum(pow(col(diff),2)).as(rmse)
    ).collect.head

    EvaluationMetrics(
      mae = metrics.getAs[Double](mae) / metrics.getAs[Long](cpt),
      rmse = Math.sqrt(metrics.getAs[Double](rmse) / metrics.getAs[Long](cpt)),
      min = metrics.getAs[Double](minCol),
      max = metrics.getAs[Double](maxCol),
      keywordPredicted = metrics.getAs[Long](cpt)
    )
  }

  def getPredictedVariableName(variableName: String): String = {
    s"${variableName}_predicted"
  }
}
