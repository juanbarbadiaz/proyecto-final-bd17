ThisBuild / version := "0.1.0-SNAPSHOT"

// Usamos Scala 2.12 o 2.13, que son las soportadas oficialmente por Spark 3.5
ThisBuild / scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
    name := "proyecto"
  )

val sparkVersion = "3.5.1"

libraryDependencies ++= Seq(
  // Librerías de Spark compatibles entre sí
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-avro" % sparkVersion,

  // Test
  "org.scalatest" %% "scalatest" % "3.2.18" % Test
)