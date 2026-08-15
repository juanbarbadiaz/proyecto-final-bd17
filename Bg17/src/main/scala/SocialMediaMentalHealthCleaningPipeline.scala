import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}

object SocialMediaMentalHealthCleaningPipeline {

  def main(args: Array[String]): Unit = {

    // 0. Inicialización de SparkSession
    val spark = SparkSession.builder()
      .appName("Spark Social Media Mental Health Cleaning")
      .master("local[*]")
      .config("spark.driver.host", "localhost")
      .config("spark.sql.warehouse.dir", "file:///C:/temp")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    // Cargar dataset original
    val path = "data/social_media_mental_health.csv"
    val rawDf = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(path)

    println("=== DATASET ORIGINAL ===")
    rawDf.show(5)

    // ==========================================
    // 1. TRATAMIENTO DE NULOS
    // ==========================================
    // 1. Reemplazar NULLs en columnas numéricas por 0.0
    val dfNumericClean = rawDf.na.fill(0.0)

    // 2. Reemplazar NULLs en columnas de texto/categóricas por "Unknown"
    val dfClean = dfNumericClean.na.fill("Unknown")

    println("\n=== FASE 1: VERIFICACIÓN DE NULOS ===")
    val totalNulls = dfClean.schema.fields.map { field =>
      dfClean.filter(col(field.name).isNull).count()
    }.sum

    println(s"-> Cantidad total de valores NULL restantes en todo el dataset: $totalNulls")

    // ==========================================
    // 2. ESTANDARIZACIÓN DE DATOS (Z-SCORE)
    // ==========================================
    // Listado de variables numéricas continuas para estandarizar
    val numericCols = Array(
      "Age", "Daily_Screen_Time_Hours", "Sleep_Duration_Hours",
      "GAD_7_Score", "PHQ_9_Score"
    )

    // Vectorizar las columnas numéricas para Spark MLlib
    val assembler = new VectorAssembler()
      .setInputCols(numericCols)
      .setOutputCol("numeric_features_unscaled")

    val vecDf = assembler.transform(dfClean)

    // Escalador Z-Score (Media = 0, Desviación Estándar = 1)
    val scaler = new StandardScaler()
      .setInputCol("numeric_features_unscaled")
      .setOutputCol("numeric_features_scaled")
      .setWithMean(true)
      .setWithStd(true)

    val scaledDf = scaler.fit(vecDf).transform(vecDf)

    println("\n=== FASE 2: DATOS NUMÉRICOS ESTANDARIZADOS (Z-SCORE) ===")
    scaledDf.select("numeric_features_scaled").show(3, false)

    // ==========================================
    // 3. DETECCIÓN Y FILTRADO DE OUTLIERS (IQR)
    // ==========================================
    def filterOutliersIQR(df: org.apache.spark.sql.DataFrame, colName: String): org.apache.spark.sql.DataFrame = {
      val quantiles = df.stat.approxQuantile(colName, Array(0.25, 0.75), 0.01)
      val q1 = quantiles(0)
      val q3 = quantiles(1)
      val iqr = q3 - q1

      val lowerBound = q1 - (1.5 * iqr)
      val upperBound = q3 + (1.5 * iqr)

      println(s"\nFiltro IQR para '$colName' -> Q1: $q1, Q3: $q3, Rango Válido: [$lowerBound, $upperBound]")
      df.filter(col(colName) >= lowerBound && col(colName) <= upperBound)
    }

    // Filtrar outliers en la variable 'Daily_Screen_Time_Hours'
    val dfFiltered = filterOutliersIQR(scaledDf, "Daily_Screen_Time_Hours")

    println("\n=== FASE 3: DATASET FILTRADO SIN OUTLIERS ===")
    println(s"-> Registros conservados tras el filtro IQR: ${dfFiltered.count()}")

    // ==========================================
    // EXPORTACIÓN A CSV LIMPIO
    // ==========================================
    println("\n=== EXPORTANDO DATASET LIMPIO A ARCHIVO CSV ===")

    // Eliminamos las columnas de vectores temporales para exportar el dataset tabular limpio
    val dfFinalToExport = dfFiltered.drop("numeric_features_unscaled", "numeric_features_scaled")

    val colsToSelect: Array[String] = dfFinalToExport.columns
    val headerClean: String = colsToSelect.mkString(",")
    val collectedRows: Array[org.apache.spark.sql.Row] = dfFinalToExport.collect()

    val fileWriter = new java.io.PrintWriter(new java.io.File("data/social_media_mental_health_clean.csv"))

    try {
      fileWriter.println(headerClean)
      for (r <- collectedRows) {
        val line = r.toSeq.map {
          case null => ""
          case cell =>
            val str = cell.toString
            if (str.contains(",")) s""""$str"""" else str
        }.mkString(",")
        fileWriter.println(line)
      }
    } finally {
      fileWriter.close()
    }

    println("\n¡PROCESO COMPLETADO AL 100%!")
    println("Dataset limpio guardado en: data/social_media_mental_health_clean.csv")

    spark.stop()
  }
}