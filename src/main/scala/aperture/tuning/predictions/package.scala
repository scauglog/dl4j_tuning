package aperture.tuning.predictions

case class FeaturesAndTarget(features: Seq[Double],
                             bid: Double,
                             target: Double)

case class CsvRow(
  key: String,
  bid: Double,
  target: Double,
  feat1: Double,
  feat2: Double,
  feat3: Double,
  feat1_1: Double,
  feat2_1: Double,
  feat3_1: Double,
  feat1_2: Double,
  feat2_2: Double,
  feat3_2: Double)

object CsvRow {
  def apply(str: String): CsvRow = {
    val row = str.split(",")
    CsvRow(
      key = row(0),
      bid = row(1).toDouble,
      target = row(2).toDouble,
      feat1 = row(3).toDouble,
      feat2 = row(4).toDouble,
      feat3 = row(5).toDouble,
      feat1_1 = row(6).toDouble,
      feat2_1 = row(7).toDouble,
      feat3_1 = row(8).toDouble,
      feat1_2 = row(9).toDouble,
      feat2_2 = row(10).toDouble,
      feat3_2 = row(11).toDouble
    )
  }
}