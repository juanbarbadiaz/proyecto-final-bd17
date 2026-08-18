import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.Imputer

object MentalWellBeing {
  def main(args: Array[String]): Unit = {

    // Inicializar SparkSession
    val spark = SparkSession.builder()
      .appName("proyecto")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    // ==========================================
    // 1. Cargar los datasets
    // ==========================================
    val dfAddiction = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/social_media_addiction_mental_wellbeing/social_media_addiction_mental_wellbeing.csv")

    //Dividir los datasets
    val weights = Array(0.8, 0.2)
    val seed = 22
    val Array(df_addiction_train, df_addiction_test) = dfAddiction.randomSplit(weights, seed)
    // ==========================================
    // 2. Comprobar Nulos
    // ==========================================
    Metodos.comprobarNulos(df_addiction_train, "Social Media Addiction")

    val totalOriginalTrain = df_addiction_train.count()
    val totalOriginalTest = df_addiction_test.count()
    //Columnas tipo
    val columnasCategoricas = Array("Gender", "Occupation", "Relationship_Status", "Primary_Platform")
    val columnasNumMWB = df_addiction_train.schema.fields
      .filter(f => f.dataType.typeName == "integer" || f.dataType.typeName == "double" || f.dataType.typeName == "long")
      .map(_.name)
      .filter(_ != "User_ID")
    val columnasYesNo = Array("Tried_To_Cut_Back", "Failed_To_Cut_Back", "Late_Night_Usage")
    //Imputamos categoricas
    val dfCategoricasImputadasTrain = df_addiction_train.na.fill("unknown", columnasCategoricas)
    val dfCategoricasImputadasTest = df_addiction_test.na.fill("unknown", columnasCategoricas)
    //Imputamos numericas
    val imputer = new Imputer()
      .setInputCols(columnasNumMWB)
      .setOutputCols(columnasNumMWB)
      .setStrategy("median")
    val imputerModel = imputer.fit(dfCategoricasImputadasTrain)
    val dfFinalImputadoTrain = imputerModel.transform(dfCategoricasImputadasTrain)
    val dfFinalImputadoTest = imputerModel.transform(dfCategoricasImputadasTest)
    //imputamos booleanas
    var dfConBooleansTrain = dfFinalImputadoTrain
    var dfConBooleansTest = dfFinalImputadoTest
    var modasMap = scala.collection.mutable.Map[String, Boolean]()
    for (columna <- columnasYesNo) {
      val modaVal = Metodos.obtenerModaBinaria(dfConBooleansTrain, columna)
      modasMap(columna) = modaVal
      println(s"La moda para $columna es: $modaVal")
      dfConBooleansTrain = dfConBooleansTrain.withColumn(
        columna,
        when(trim(lower(col(columna))) === "yes", true)
          .when(trim(lower(col(columna))) === "no", false)
          .otherwise(modaVal) // Imputa la moda calculada (1 o 0) si es nulo
      )
    }
    for (columna <- columnasYesNo) {
      val modaValTrain = modasMap(columna) // Recuperamos la moda exacta que se usó en Train

      dfConBooleansTest = dfConBooleansTest.withColumn(
        columna,
        when(trim(lower(col(columna))) === "yes", true)
          .when(trim(lower(col(columna))) === "no", false)
          .otherwise(modaValTrain) // Usamos la moda del train, NUNCA una nueva calculada en test
      )
    }
    val dfLimpioTrain = dfConBooleansTrain.na.drop()
    val dfLimpioTest = dfConBooleansTest.na.drop()

    // 4. Obtener el número final de filas tras la limpieza
    val totalFinalTrain = dfLimpioTrain.count()
    val totalFinalTest = dfLimpioTest.count()
    // 5. Calcular filas eliminadas y el porcentaje de pérdida
    val filasEliminadasTrain = totalOriginalTrain - totalFinalTrain
    val filasEliminadasTest = totalOriginalTest - totalFinalTest
    val porcentajePerdidoTrain = (filasEliminadasTrain.toDouble / totalOriginalTrain) * 100
    val porcentajePerdidoTest = (filasEliminadasTest.toDouble / totalOriginalTest) * 100

    // 6. Mostrar métricas de la limpieza
    println("\n==============================================")
    println("        INFORME DE LIMPIEZA DE DATOS          ")
    println("==============================================")
    println(s"Filas originales Train:           $totalOriginalTrain")
    println(s"Filas originales Test:           $totalOriginalTest")
    println(s"Filas conservadas Train:          $totalFinalTrain")
    println(s"Filas conservadas Test:          $totalFinalTest")
    println(s"Filas eliminadas Train:           $filasEliminadasTrain")
    println(s"Filas eliminadas Test:           $filasEliminadasTest")
    println(f"Porcentaje de datos perdidos Train: $porcentajePerdidoTrain%.2f%%")
    println(f"Porcentaje de datos perdidos Test: $porcentajePerdidoTest%.2f%%")

    println("==============================================\n")

    // Mostrar las primeras filas del dataset limpio
    dfLimpioTrain.show(5)

    // ==========================================
    // 3. Estandarizar Datos
    // ==========================================
    val dfFinalFormateadoMWBTrain = Metodos.estandarizarDatos(dfLimpioTrain)
    val dfFinalFormateadoMWBTest = Metodos.estandarizarDatos(dfLimpioTest)
    println("--- Muestra del Dataset Final Formateado ---")
    dfFinalFormateadoMWBTrain.show(5)
    // ==========================================
    // 4. Outliers
    // ==========================================

    val limitesIQR = Metodos.calcularLimitesIQR(dfFinalFormateadoMWBTrain,2)
    val dfFinalMWBTrain = Metodos.aplicarLimitesIQR(dfFinalFormateadoMWBTrain, limitesIQR)
    val dfFinalMWBTest = Metodos.aplicarLimitesIQR(dfFinalFormateadoMWBTest,limitesIQR)
    //Descarga de Datasets procesados
    dfFinalMWBTrain
      .coalesce(1)
      .write
      .option("header", "true")
      .option("sep", ",")
      .mode("overwrite")
      .csv("output/dfFinalMWB_Train")

    dfFinalMWBTest
      .coalesce(1)
      .write
      .option("header", "true")
      .option("sep", ",")
      .mode("overwrite")
      .csv("output/dfFinalMWB_Test")

    println("¡Datasets guardados con éxito en la carpeta 'output/'!")
    spark.stop()
  }
}
