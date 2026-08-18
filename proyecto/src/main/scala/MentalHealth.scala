import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object MentalHealth {
  def main(args: Array[String]): Unit = {
    // Inicializar SparkSession
    val spark = SparkSession.builder()
      .appName("proyecto")
      .master("local[*]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val dfMentalHealth = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/social_media_mental_health/social_media_mental_health.csv")

    //Dividir los datasets
    val weights = Array(0.8, 0.2)
    val seed = 22
    val Array(df_ma_train, df_ma_test) = dfMentalHealth.randomSplit(weights, seed)
    //Comprobar nulos
    Metodos.comprobarNulos(df_ma_train, "Social Media Mental Health")
    // Estandarizar datos
    val columnasBooleanas = Array("Late_Night_Usage", "Social_Comparison_Trigger")
    var dfConBooleansMATrain = df_ma_train
    var dfConBooleansMATest = df_ma_test
    for (columna <- columnasBooleanas) {
      dfConBooleansMATrain = dfConBooleansMATrain.withColumn(
        columna,
        when(trim(lower(col(columna))).isin("1", "true"), true)
          .when(trim(lower(col(columna))).isin("0", "false"), false)
          .otherwise(null)
      )
    }
    for (columna <- columnasBooleanas) {
      dfConBooleansMATest = dfConBooleansMATest.withColumn(
        columna,
        when(trim(lower(col(columna))).isin("1", "true"), true)
          .when(trim(lower(col(columna))).isin("0", "false"), false)
          .otherwise(null)
      )
    }
    val dfFinalFormateadoMATrain = Metodos.estandarizarDatos(dfConBooleansMATrain)
    val dfFinalFormateadoMATest = Metodos.estandarizarDatos(dfConBooleansMATest)
    println("--- Muestra del Dataset Final Formateado ---")
    dfFinalFormateadoMATrain.show(5)

    //Eliminar Outliers
    val limitesIQR = Metodos.calcularLimitesIQR(dfFinalFormateadoMATrain, 1.5)
    val dfFinalMATrain = Metodos.aplicarLimitesIQR(dfFinalFormateadoMATrain, limitesIQR)
    val dfFinalMATest = Metodos.aplicarLimitesIQR(dfFinalFormateadoMATest,limitesIQR)
    // 2. Guardar dfFinalMA como CSV en una carpeta local
    dfFinalMATrain
      .coalesce(1)
      .write
      .option("header", "true")
      .option("sep", ",")
      .mode("overwrite")
      .csv("output/dfFinalMA_Train")

    dfFinalMATest
      .coalesce(1)
      .write
      .option("header", "true")
      .option("sep", ",")
      .mode("overwrite")
      .csv("output/dfFinalMA_Test")

    println("¡Datasets guardados con éxito en la carpeta 'output/'!")

    spark.stop()
  }
}
