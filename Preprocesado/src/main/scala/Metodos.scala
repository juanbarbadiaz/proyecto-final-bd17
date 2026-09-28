import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, desc, lower, round, sum, trim, upper}
import org.apache.spark.sql.types.{DoubleType, FloatType, NumericType, StringType}
import org.apache.hadoop.fs.{FileSystem, Path}

object Metodos {
  def comprobarNulos(df: DataFrame, nombre: String): Unit = {
    println(s"\n--- Recuento de nulos para: $nombre ---")
    val exprs = df.columns.map(c => sum(col(c).isNull.cast("int")).alias(c))
    df.select(exprs: _*).show()
  }
  def obtenerModaBinaria(df: DataFrame, columna: String): Boolean = {
    val valorModa = df
      .filter(col(columna).isNotNull)
      .groupBy(trim(lower(col(columna))).alias("valor"))
      .count()
      .orderBy(desc("count"))
      .first()
      .getString(0) // Devuelve "yes" o "no"

    if (valorModa == "yes") true else false
  }
  def estandarizarDatos(df: DataFrame): DataFrame = {
    val colsTexto = df.schema.fields
      .filter(_.dataType == StringType)
      .map(_.name)

    // Detectar dinámicamente TODAS las columnas decimales
    val colsDecimales = df.schema.fields
      .filter(f => f.dataType == DoubleType || f.dataType == FloatType)
      .map(_.name)

    var dfFormateado = df

    // 3. Aplicar mayúsculas y limpiar espacios extra en columnas de texto
    for (columna <- colsTexto) {
      dfFormateado = dfFormateado.withColumn(columna, upper(trim(col(columna))))
    }

    // 4. Redondear a 1 decimal todas las columnas numéricas continuas
    for (columna <- colsDecimales) {
      dfFormateado = dfFormateado.withColumn(columna, round(col(columna), 1))
    }
    dfFormateado.show(5)
    dfFormateado
  }

  def calcularLimitesIQR(df: DataFrame, factor: Double): Map[String, (Double, Double)] = {
    val columnasNumericas = df.schema.fields
      .filter(_.dataType.isInstanceOf[NumericType])
      .map(_.name)

    columnasNumericas.map { columna =>
      val quantiles = df.stat.approxQuantile(columna, Array(0.25, 0.75), 0.01)

      val q1 = quantiles(0)
      val q3 = quantiles(1)
      val iqr = q3 - q1

      val limiteInferior = q1 - factor * iqr
      val limiteSuperior = q3 + factor * iqr

      columna -> (limiteInferior, limiteSuperior)
    }.toMap
  }
  def aplicarLimitesIQR(df: DataFrame, limites: Map[String, (Double, Double)]): DataFrame = {
    val filasIniciales = df.count()
    val dfFiltrado = limites.foldLeft(df) {
      case (dfActual, (columna, (limiteInferior, limiteSuperior))) =>
        dfActual.filter(
          col(columna).between(limiteInferior, limiteSuperior)
        )
    }
    // Métricas finales
    val filasFinales = dfFiltrado.count()
    val filasEliminadas = filasIniciales - filasFinales
    val porcentajeMantenido =
      (filasFinales.toDouble / filasIniciales) * 100
    val porcentajeEliminado =
      (filasEliminadas.toDouble / filasIniciales) * 100

    println("\n----------------------------------------------")
    println(s"Filas iniciales:                      $filasIniciales")
    println(s"Filas eliminadas por outliers:        $filasEliminadas (${f"$porcentajeEliminado%.2f"}%%)")
    println(s"Filas MANTENIDAS (Dataset limpio):    $filasFinales (${f"$porcentajeMantenido%.2f"}%%)")
    println("==============================================\n")
    dfFiltrado
  }

  def guardarCsvConNombre(df: org.apache.spark.sql.DataFrame, folderPath: String, fileName: String): Unit = {
    val fs = FileSystem.get(df.sparkSession.sparkContext.hadoopConfiguration)

    val finalDir = new Path(folderPath)
    val tempDir = new Path(s"${folderPath}_temp")
    val targetFile = new Path(s"$folderPath/$fileName.csv")

    // 1. Limpiar carpeta temporal si ya existía
    if (fs.exists(tempDir)) fs.delete(tempDir, true)

    // 2. Escribir en la carpeta temporal
    df.coalesce(1)
      .write
      .option("header", "true")
      .option("sep", ",")
      .mode("overwrite")
      .csv(tempDir.toString)

    // 3. Buscar el archivo part-00000-...csv
    val partFile = fs.listStatus(tempDir)
      .map(_.getPath)
      .find(_.getName.startsWith("part-"))
      .getOrElse(throw new Exception("No se encontró el archivo CSV generado"))

    // 4. Limpiar la carpeta final y recrearla limpia
    if (fs.exists(finalDir)) fs.delete(finalDir, true)
    fs.mkdirs(finalDir)

    // 5. Mover el archivo CSV con su nuevo nombre y borrar la carpeta temporal
    fs.rename(partFile, targetFile)
    fs.delete(tempDir, true)
  }
}
