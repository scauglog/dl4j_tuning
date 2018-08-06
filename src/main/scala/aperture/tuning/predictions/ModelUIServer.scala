package aperture.tuning.predictions

import org.deeplearning4j.ui.api.UIServer

object ModelUIServer {
  def main(args: Array[String]): Unit = {
    val uiServer: UIServer = UIServer.getInstance
    uiServer.enableRemoteListener()
  }
}
