ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "2.12.15"

// Repositorios explícitos para asegurar descargas
resolvers ++= Seq(
  "Maven Central" at "https://repo1.maven.org/maven2/",
  "Spark Packages Repo" at "https://repos.spark-packages.org/"
)

val sparkVersion = "3.5.0"

lazy val root = (project in file("."))
  .settings(
    name := "Bg17",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql"  % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion
    )
  )