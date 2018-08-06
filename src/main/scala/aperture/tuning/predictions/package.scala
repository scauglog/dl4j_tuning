package aperture.tuning.predictions

case class FeaturesAndTarget(features: Seq[Double],
                             bid: Double,
                             target: Double)