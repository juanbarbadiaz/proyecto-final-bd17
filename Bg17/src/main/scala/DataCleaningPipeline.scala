import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder}
import org.apache.spark.ml.Pipeline

object DataCleaningPipeline {

  def main(args: Array[String]): Unit = {

    // =========================================================================
    // 0. CONFIGURACIÓN E INICIALIZACIÓN DE SPARK SESSION
    // =========================================================================
    val spark = SparkSession.builder()
      .appName("Spark Big Data Cleaning & Dual Export")
      .master("local[*]")
      .config("spark.driver.host", "localhost")
      .config("spark.sql.warehouse.dir", "file:///C:/temp")
      .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    // Carga del dataset original
    val path = "data/social_media_addiction_mental_wellbeing.csv"
    val rawDf = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(path)

    println("=== DATASET ORIGINAL ===")
    rawDf.show(5)

    // =========================================================================
    // FASE 1: TRATAMIENTO DE NULOS
    // =========================================================================
    // Reemplazar NULLs numéricos por 0.0 y categóricos por "Unknown"
    val dfNumericClean = rawDf.na.fill(0.0)
    val dfClean = dfNumericClean.na.fill("Unknown")

    println("\n=== FASE 1 COMPLETADA: VERIFICACIÓN DE NULOS ===")
    val totalNulls = dfClean.schema.fields.map { field =>
      dfClean.filter(col(field.name).isNull).count()
    }.sum

    println(s"-> Cantidad total de valores NULL restantes en todo el Dataset: $totalNulls")

    // =========================================================================
    // FASE 2: ESTANDARIZACIÓN DE DATOS (Z-SCORE)
    // =========================================================================
    val numericCols = Array(
      "Daily_Usage_Hours", "FOMO_Score", "Social_Comparison_Score",
      "Validation_Seeking_Score", "Anxiety_Score", "Depression_Score",
      "Loneliness_Score", "Self_Esteem_Score", "Sleep_Quality_Score",
      "Sleep_Hours", "Productivity_Loss_Score", "Offline_Relationship_Quality",
      "Physical_Activity_Hrs_Week", "Screen_Free_Time_Hrs"
    )

    val assembler = new VectorAssembler()
      .setInputCols(numericCols)
      .setOutputCol("numeric_features_unscaled")

    val vecDf = assembler.transform(dfClean)

    val scaler = new StandardScaler()
      .setInputCol("numeric_features_unscaled")
      .setOutputCol("numeric_features_scaled")
      .setWithMean(true)
      .setWithStd(true)

    val scaledDf = scaler.fit(vecDf).transform(vecDf)

    println("\n=== FASE 2 COMPLETADA: VARIABLES NUMÉRICAS ESTANDARIZADAS ===")
    scaledDf.select("numeric_features_scaled").show(3, false)

    // =========================================================================
    // FASE 3: DETECCIÓN Y FILTRADO DE OUTLIERS (IQR)
    // =========================================================================
    def filterOutliersIQR(df: org.apache.spark.sql.DataFrame, colName: String): org.apache.spark.sql.DataFrame = {
      val quantiles = df.stat.approxQuantile(colName, Array(0.25, 0.75), 0.01)
      val q1 = quantiles(0)
      val q3 = quantiles(1)
      val iqr = q3 - q1

      val lowerBound = q1 - (2 * iqr)
      val upperBound = q3 + (2 * iqr)

      println(s"\nFiltro IQR para '$colName' -> Q1: $q1, Q3: $q3, Rango Válido: [$lowerBound, $upperBound]")
      df.filter(col(colName) >= lowerBound && col(colName) <= upperBound)
    }

    val dfFiltered = filterOutliersIQR(scaledDf, "Daily_Usage_Hours")

    // =========================================================================
    // FASE 4: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
    // =========================================================================
    println("\n=== FASE 4: ESTADÍSTICAS DESCRIPTIVAS ===")
    dfFiltered.select("Daily_Usage_Hours", "Anxiety_Score", "Mental_Wellbeing_Score")
      .summary("count", "mean", "stddev", "min", "50%", "max")
      .show()

    println("=== AGRUPACIÓN: BIENESTAR SEGÚN NIVEL DE ADICCIÓN ===")
    dfFiltered.groupBy("Addiction_Level")
      .agg(
        avg("Daily_Usage_Hours").as("Avg_Daily_Hours"),
        avg("Mental_Wellbeing_Score").as("Avg_Mental_Wellbeing"),
        count("User_ID").as("Total_Users")
      )
      .orderBy(desc("Avg_Daily_Hours"))
      .show()

    // =========================================================================
    // FASE 5: FEATURE ENGINEERING PARA MACHINE LEARNING
    // =========================================================================
    val categoricalCols = Array(
      "Gender", "Occupation", "Relationship_Status", "Primary_Platform",
      "Late_Night_Usage", "First_Check_Morning", "Tried_To_Cut_Back", "Failed_To_Cut_Back"
    )

    val indexerOutputs = categoricalCols.map(_ + "_Index")
    val stringIndexer = new StringIndexer()
      .setInputCols(categoricalCols)
      .setOutputCols(indexerOutputs)

    val encoderOutputs = categoricalCols.map(_ + "_Vec")
    val oneHotEncoder = new OneHotEncoder()
      .setInputCols(indexerOutputs)
      .setOutputCols(encoderOutputs)

    val labelIndexer = new StringIndexer()
      .setInputCol("Addiction_Level")
      .setOutputCol("label")

    val finalAssembler = new VectorAssembler()
      .setInputCols(Array("numeric_features_scaled") ++ encoderOutputs)
      .setOutputCol("features")

    val mlPipeline = new Pipeline()
      .setStages(Array(stringIndexer, oneHotEncoder, labelIndexer, finalAssembler))

    val pipelineModel = mlPipeline.fit(dfFiltered)
    val dfMLReady = pipelineModel.transform(dfFiltered)

    // =========================================================================
    // FASE 6A: EXPORTACIÓN 1 -> DATASET LIMPIO PARA ANÁLISIS / EXCEL
    // =========================================================================
    println("\n=== GENERANDO ARCHIVO 1: social_media_clean.csv ===")

    val dfCleanExport = dfFiltered.drop("numeric_features_unscaled", "numeric_features_scaled")
    val colsClean: Array[String] = dfCleanExport.columns
    val headerClean: String = colsClean.mkString(",")
    val rowsClean: Array[org.apache.spark.sql.Row] = dfCleanExport.collect()

    val writerClean = new java.io.PrintWriter(new java.io.File("data/social_media_clean.csv"))

    try {
      writerClean.println(headerClean)
      for (r <- rowsClean) {
        val line = r.toSeq.map {
          case null => ""
          case cell =>
            val str = cell.toString
            if (str.contains(",")) s""""$str"""" else str
        }.mkString(",")
        writerClean.println(line)
      }
    } finally {
      writerClean.close()
    }

    println("-> ¡Archivo 1 guardado con éxito!: data/social_media_clean.csv")

    // =========================================================================
    // FASE 6B: EXPORTACIÓN 2 -> DATASET NUMÉRICO 100% LISTO PARA ML
    // =========================================================================
    println("\n=== GENERANDO ARCHIVO 2: social_media_ml_ready.csv ===")

    // Usamos selectExpr o selección de columnas tipada explícitamente
    val selectCols = Array("label") ++ numericCols ++ indexerOutputs
    val dfMLExport = dfMLReady.select(selectCols.map(col): _*)

    val colsML: Array[String] = dfMLExport.columns
    val headerML: String = colsML.mkString(",")
    val rowsML: Array[org.apache.spark.sql.Row] = dfMLExport.collect()

    val writerML = new java.io.PrintWriter(new java.io.File("data/social_media_ml_ready.csv"))

    try {
      writerML.println(headerML)
      for (r <- rowsML) {
        val line = r.toSeq.map {
          case null => "0"
          case cell => cell.toString
        }.mkString(",")
        writerML.println(line)
      }
    } finally {
      writerML.close()
    }

    println("-> ¡Archivo 2 guardado con éxito!: data/social_media_ml_ready.csv")

    println("\n=======================================================")
    println("¡PROCESO FINALIZADO EXITOSAMENTE Y SIN ERRORES!")
    println("1. data/social_media_clean.csv    -> Datos limpios legibles")
    println("2. data/social_media_ml_ready.csv -> Datos 100% numéricos para ML")
    println("=======================================================")

    spark.stop()
  }
}