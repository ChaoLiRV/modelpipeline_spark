# Create and Deploy a Machine Learning Model Pipeline in Spark

As data scientists in Red Ventures, we define ourselves as [type _'B'_ data scientist]( <https://www.dezyre.com/article/type-a-data-scientist-vs-type-b-data-scientist/194>), which differentiates with the role of business intelligents or data analysts. Besides providing business insights to drive actionable items, we are also dedicated to building models from end to end and deploying them as a service for business usage, both internally and externally . This post describes the general process of building a classification model pipeline in **Spark** and touches upon its deployment via a REST API.

_The Spark code in this post is written in **Scala** and run on the **databricks** platform_

## Data Import

The data used for training the model is stored in the Snowflake warehouse on the cloud. The [databricks-snowflake connector](https://docs.databricks.com/spark/latest/data-sources/snowflake.html) enables us to run SQL query to read data from Snowflake database. The `readDatafromSnowflake` function below configures the connection with Snowflake and executes the query of selecting all data fields from the Snowflake table `myTable` under the schema `mySchema` for a particular date range. The start and end date of the training set is specifed by the function arguments, and the function returns the entire dataset in a Dataframe format.
```scala
import net.snowflake.spark.snowflake.SnowflakeConnectorUtils
import net.snowflake.spark.snowflake.Utils.SNOWFLAKE_SOURCE_NAME
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._

def readDatafromSnowflake(dateStart: String, dateStop: String) : DataFrame = {
  // dateStart:String e.g. "2017-12-01"
  // dateStop:String e.g. "2018-06-01"
  
  // Snowflake configuration
  val user = dbutils.secrets.get("data-warehouse", "snowflake-user") // encode the username
  val password = dbutils.secrets.get("data-warehouse", "snowflake-password")// encode the password
  val SNOWFLAKE_SOURCE_NAME = "net.snowflake.spark.snowflake"
  val sfOptions = Map(
    "sfURL" -> "URLName",
    "sfAccount" -> "AccountName",
    "sfUser" -> user,
    "sfPassword" -> password,
    "sfDatabase" -> "DataBaseName",
    "sfSchema" -> "SchemaName",
    "sfWarehouse" -> "WarehouseName",
    "sfRole" -> "RoleName"
  ) 
  //  SQL query 
  val query_string = s"""
    select *
    from mySchema.myTable
    where date between '$dateStart' and '$dateStop'
  """
  //Execute query to select data from snowflake db.
  val df: DataFrame = sqlContext.read
    .format(SNOWFLAKE_SOURCE_NAME)
    .options(sfOptions)
    .option("query", query_string)
    .load()
  df
}
```
In order to automate the model update process, the date range is extracted from the system datetime every time the job is running. The code below implements the function to obtain the training data from a six month (180 days) look-back window. A snapshot of this data set suggests the features are comprised of various types, such as categorical variables _Var1, Var2, Var3_, binary variable _Var4_, and numeric variable types _Var5, Var6_. Here _label_ is the classification target that the model learns to predict. 
```scala
import java.util.{Calendar, Date}
import java.text.SimpleDateFormat 

val formatter = new SimpleDateFormat("YYYY-MM-dd")
val calendar = Calendar.getInstance()
val dateStop:String = formatter.format(calendar.getTime()) // get current date
calendar.add(Calendar.DAY_OF_YEAR, -180) // look back over the past 180 days
val dateStart:String = formatter.format(calendar.getTime())

var df_all :DataFrame = readDatafromSnowflake(dateStart, dateStop)
display(df_all.limit(5))
```
| ID        | Var1       | Var2          | Var3     | Var4 | Var5      | Var6 | label |
|-----------|------------|---------------|----------|------|-----------|------|-------|
| 0001 | SmartPhone | Mobile Safari | Off-Peak | 1    | 484500.47 | 16   | 0     |
| 0002 | Desktop    | Firefox       | Peak     | 0    | 513500.32 | NaN   | 1     |
| 0003 | Desktop    | Chrome        | Off-Peak | 0    | 441000.45 | 25   | 0     |
| 0004 | Desktop    | Safari        | Weekend  | 0    | 840000.4  | 2    | 0     |
| 0005 | Desktop    | Chrome        | Peak     | 1    | NaN       | 22   | 1     |

It is obvious that the _Var1-3_ represent the device type, browser type, and time range respectively. To make these variable names descriptive, you can rename the dataframe columns using Spark Dataframe's method `WithColumnRenamed()`. In this exercise, I found a really useful function [foldLeft()](http://allaboutscala.com/tutorials/chapter-8-beginner-tutorial-using-scala-collection-functions/scala-foldleft-example/) that makes this process more scalable, in the sense that you need not write 100 statements for 100 variable's name changes. All you need is to make the _old name_ and _new name_ as a key-value pair and store it in the scala _Map_ value. No other code change is required. Similary, another example demonstrates how foldLeft is employed to replace NaN value for each numeric variable by its mean or median in the training data.
```scala
/* rename column names */
// df_all = df_all.withColumnRenamed("Var1", "Device")  
// df_all = df_all.withColumnRenamed("Var2", "Browser") 
// df_all = df_all.withColumnRenamed("Var3", "TimeRange") 

/* rename column names using foldLeft */
val columnNameMap = Map("Var1"->"Device",
                        "Var2"->"Browser", 
                        "Var3"->"TimeRange")
df_all = columnNameMap.keys.foldLeft(df_all){
  (tmpDF, colName) => 
  tmpDF.withColumnRenamed(colName, columnNameMap(colName) ) 
}

/* fill NaN values of each numeric variable with column mean */
val colNames = Array("Var5", "Var6")
val colMeans = colNames.map(colName => (colName, df_all.select(avg(col(colName))).first.getDouble(0)) ).toMap // Map(colname->colMean)  
// val colMedian = colNames.map(colName => (colName, df_all.select(col(colName)).orderBy(desc(colName)).limit((df_all.count()/2).toInt).orderBy(asc(colName)).first().getDouble(0)) ).toMap //Map(colname->colMedian)      
val DUMMY_NULL = -1.0
df_all = df_all.na.fill(DUMMY_NULL, colNames)
df_all = colNames.foldLeft(df_all){
  (tmpDF, colName) => tmpDF.withColumn(colName, 
                     when(col(colName)===DUMMY_NULL, colMeans(colName)).otherwise(col(colName)) ) 
}
display(df_all.limit(5))
```

| ID   | Device     | Browser       | TimeRange| Var4 | Var5      | Var6 | label |
|------|------------|---------------|----------|------|-----------|------|-------|
| 0001 | SmartPhone | Mobile Safari | Off-Peak | 1    | 484500.47 | 16   | 0     |
| 0002 | Desktop    | Firefox       | Peak     | 0    | 513500.32 | NaN   | 1     |
| 0003 | Desktop    | Chrome        | Off-Peak | 0    | 441000.45 | 25   | 0     |
| 0004 | Desktop    | Safari        | Weekend  | 0    | 840000.4  | 2    | 0     |
| 0005 | Desktop    | Chrome        | Peak     | 1    | NaN       | 22   | 1     |


Since the subject of this article is about building a model pipeline for deployment, the data exploration section is done off-line and not described here. Next it comes to the section of pipeline creation for feature engineering and model training.
```scala
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.attribute._

// Split traing- testing dataset
val splitSeed = 1234
val Array(trainingData, testingData) = df_all.randomSplit(Array(0.8, 0.2), splitSeed)
```

## Feature Transformer Pipeline 

#### Numeric Variables
For model running in production, it is always a good habit of setting a defensive layer to handle the anomaly gracefully. In this example, we have an **Imputer** transformer first in the pipeline to handle the missing values for numeric variables. The choice of the transformer is to some extent limited to the availability in **MLeap** _(refer to this document: <http://mleap-docs.combust.ml/core-concepts/transformers/support.html>)_. MLeap is the tool that we use to serialize the Spark model pipelines and we will touch on that later in this post. Suppose the transfomer function that you need does not exist on their list, follow the procedure here <http://mleap-docs.combust.ml/mleap-runtime/custom-transformer.html> to create the custom transformer.
```scala
import org.apache.spark.ml.mleap.feature.Imputer

// Handle missing value of numeric variables
val var5Imputer = new Imputer()
  .setInputCol("Var5")
  .setOutputCol("Var5Impute")
  .setStrategy("median")
val var6Imputer = new Imputer()
  .setInputCol("Var6")
  .setOutputCol("Var6Impute")
  .setStrategy("median")
```
| ID        | Device     | Browser       | TimeRange | Var4 | Var5      | Var6 | label | Var5Impute | Var6Impute |
|-----------|------------|---------------|-----------|------|-----------|------|-------|------------|------------|
| 136581531 | SmartPhone | Mobile Safari | Off-Peak  | 1    | 484500.47 | 16   | 0     | 484500.47  | 16         |
| 138079502 | Desktop    | Firefox       | Peak      | 0    | 513500.32 | NaN  | 1     | 513500.32  | 16         |
| 136280501 | Desktop    | Chrome        | Off-Peak  | 0    | 441000.45 | 25   | 0     | 441000.45  | 25         |
| 136608744 | Desktop    | Safari        | Weekend   | 0    | 840000.4  | 2    | 0     | 840000.4   | 2          |
| 136576205 | Desktop    | Chrome        | Peak      | 1    | NaN       | 22   | 1     | 275500.5   | 22         |

Next we perform numeric operations on these variables, such as dividing one feature value by the other, taking the logarithmic transform of the value, and scale normalization (Min-Max or Z-score).
```scala
import org.apache.spark.ml.mleap.feature.MathBinary
import ml.combust.mleap.core.feature.MathBinaryModel
import ml.combust.mleap.core.feature.BinaryOperation._
import org.apache.spark.ml.mleap.feature.MathUnary
import ml.combust.mleap.core.feature.MathUnaryModel
import ml.combust.mleap.core.feature.UnaryOperation._

// binary operation (division) of two data features
val divider = new MathBinary(uid = "loantovalue", model = MathBinaryModel(Divide)) // 
                .setInputA("Var5Impute")
                .setInputB("Var6Impute")
                .setOutputCol("var5_to_var6")
// unary operation (log transformation) on a single feature
val logTramsformer = new MathUnary(uid = "ltvlog", model = MathUnaryModel(Log)) // 
                .setInputCol("Var5Impute")
                .setOutputCol("var_log")

val assemblerNum = new VectorAssembler().setInputCols(Array("var5_to_var6","var_log"))
                                        .setOutputCol("numeric_features_vec")
val scaler = new feature.StandardScaler()
                .setInputCol("numeric_features_vec")
                .setOutputCol("scaled_numeric_features")
                .setWithStd(true)
                .setWithMean(true)
```
#### Categorical Variables
The first process step is same as the numerical variables, to set an imputation stage for handling the missing value in real world scenario. MLeap does not provide this transformer function as you cannot find it on this list <http://mleap-docs.combust.ml/core-concepts/transformers/support.html>, and therefore we create our own transformer _**StringImputer**_ by following the MLeap document as aforementioned. In categorical variables, sometimes values are representing the same thing and can be bucketed into one group, for instance _"Mobile"_ and _"SmartPhone"_. In this situation, the _**StringMapper**_ transformer is employed to achieve this. Note that I utilize a custom transformer in the code instead of the MLeap built-in _StringMap_, for the reason that their transformer does not allow to set the default value in the map. Next stage in the pipeline is the _**StringIndexer**_ that is another defensive layer to handle unseen values during training.
```scala
import com.redventures.custom.core.transformer.StringMapperModel 
import org.apache.spark.ml.custom.transformer.StringMapper
import org.apache.spark.ml.custom.transformer.StringImputer

/* Device */
val deviceImputer = new StringImputer(uid = "device_imp", model = StringImputerModel("OtherDevices")) 
                .setInputCol("Device")
                .setOutputCol("device_imp")
val deviceMapper = new StringMapper(uid = "device_map", model = StringMapperModel(
                Map("Mobile"->"SmartPhone","SmartPhone"->"SmartPhone","Desktop"->"Desktop","Tablet"->"Tablet"), "OtherDevices") ) //
                .setInputCol("device_imp")
                .setOutputCol("device_mod")
val deviceIndexer = new feature.StringIndexer()
                .setInputCol("device_mod")
                .setOutputCol("deviceTypeIndex")
                .setHandleInvalid("keep")
```
Repeat the procedures to apply transformers to other categorical variables _TimeRange_ and _Browser_. And then set the _one-hot encoding_ stage for all processed categorical variables.
```scala
/* TimeRange */
val timeImputer = new StringImputer(uid = "time_imp", model = StringImputerModel("Peak"))
                .setInputCol("TimeRange")
                .setOutputCol("Time_imp")
val timeRangeIndexer = new StringIndexer()
                .setInputCol("Time_imp")
                .setOutputCol("timeRangeIndex")
                .setHandleInvalid("keep")
/* Browser */
val browserImputer = new StringImputer(uid = "browser_imp", model = StringImputerModel("OtherBrowsers"))
                .setInputCol("Browser")
                .setOutputCol("browser_imp")
val browserIndexer = new StringIndexer()
                .setInputCol("browser_imp")
                .setOutputCol("browserIndex")
                .setHandleInvalid("keep")

/* One hot encoding */
val ohcs = new OneHotEncoderEstimator()
              .setInputCols(Array("deviceTypeIndex","timeRangeIndex","browserIndex"))
              .setOutputCols(Array("deviceTypeVec","timeRangeVec","browserVec"))
              .setDropLast(true)
```
The classification model in **spark.ml** package is an estimator, which requires all feature data assembled as a vector for each record in the column _`"feature"`_. This is done by the transformer `VectorAssembler()`. Lastly, we stack all the aforementioned transformers in the sequence to a pipeline object. As a consequence, every single record of both training and testing set is guaranteed to go through the same feature engineering process without incurring exception by anomaly value.
```scala
/* assemble all processed features into a single vector */
val featureCols = Array("deviceTypeVec", "timeRangeVec", "browserVec", "Var4", "scaled_numeric_features")
val assembler_all = new feature.VectorAssembler().setInputCols(featureCols).setOutputCol("features")

// set the pipeline to include all the feature engineering stages
val feature_stages = new Pipeline()
                .setStages( Array(var5Imputer, var6Imputer, // numeric variable imputer
                                  divider, logTramsformer, assembler_num, scaler,  //numeric variable processing
                                  deviceImputer, deviceMapper, deviceIndexer, // Device type processing
                                  timeImputer, timeRangeIndexer, // Time range
                                  browserImputer, browserIndexer, // Browser 
                                  ohcs, // one hot encoding
                                  assembler_all))
```

## Model Estimator

In this example, the gradient boosting classifier _GBTClassifier_ is chosen to do the predictive task as it achieves state-of-the-art performance. The _CrossValidator_ is employed for the model tuning and the _Parallelizable_ version of cross validator can speed up the computation. Once the model parameters are determined by the tuning, the model estimator is appended to the pipeline as the last stage. The pipeline object's _`fit`_ methond executes the entire workflow including both the feature engineering and model training process on the dataset.
```scala
var gbt = new classification.GBTClassifier().setSeed(1234)

// add estimator to Pipeline
val pipeline_cv = new Pipeline().setStages(feature_stages.getStages ++ Array(gbt))

// cross validation of pipeline
val gbtParamGrid = new tuning.ParamGridBuilder()
                .addGrid(gbt.maxDepth, Array(3, 4, 5))
                .addGrid(gbt.stepSize, Array(0.001, 0.01, 0.1))
                .addGrid(gbt.maxIter, Array(100, 200, 300))   
                .build()
val gbtEvaluator = new evaluation.BinaryClassificationEvaluator().setMetricName("areaUnderROC")
val parallelismWorkers:Int = 8 // number of works
val gbtCV = new tuning.ParallelizableCrossValidator()
            .setEstimator(pipeline_cv)
            .setEstimatorParamMaps(gbtParamGrid)
            .setEvaluator(gbtEvaluator)
            .setNumFolds(5)
            .setParallelism(parallelismWorkers)

val model_cv = gbtCV.fit(trainingData)
val bestModel = model_cv.bestModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[classification.GBTClassificationModel]

val gbt_best = gbt.setMaxDepth(bestModel.getMaxDepth).setStepSize(bestModel.getStepSize).setMaxIter(bestModel.getMaxIter)

// add model training stage to complete the pipeline 
val final_pipeline_gbt = new Pipeline().setStages(feature_stages.getStages ++ Array(gbt_best))
val model_gbt = final_pipeline_gbt.fit(trainingData)
```
#### Model Evaluation
Since this is a binary classification task, the area under the _ROC_ curve is calculated to measure the model performance. In addition, the gradient boosting model are also evaluated against the vanilla version model logistic regression.
```scala
// benchmark model performance
var lr = new classification.LogisticRegression().setMaxIter(100).setElasticNetParam(1).setRegParam(0.001)
val final_pipeline_lr = new Pipeline().setStages(feature_stages.getStages ++ Array(lr))
val model_lr = final_pipeline_lr.fit(trainingData)

// Model comparison in terms of AUC value
val predictions_gbt = model_gbt.transform(testingData)
val predictions_lr = model_lr.transform(testingData)
val scores_gbt = new evaluation.BinaryClassificationEvaluator().setMetricName("areaUnderROC").evaluate(predictions_gbt)
val scores_lr = new evaluation.BinaryClassificationEvaluator().setMetricName("areaUnderROC").evaluate(predictions_lr)
println(scores_gbt, scores_lr)
```
Logistic regression is considered as a white box algorithm. The code below extracts the coefficients for the model.
```scala
// Print Logistic Regression coefficients
val schema = predictions_lr.schema
// Extract the attributes of the input (features) column to our logistic regression model
val pipeline_model_lr = model_lr.asInstanceOf[PipelineModel].stages.last.asInstanceOf[classification.LogisticRegressionModel]
val featureAttrs = AttributeGroup.fromStructField(schema(pipeline_model_lr.getFeaturesCol)).attributes.get
val features = featureAttrs.map(_.name.get)
val featureNames: Array[String] = if (pipeline_model_lr.getFitIntercept) {
  Array("(Intercept)") ++ features
} else {
  features
}
val lrModelCoeffs = pipeline_model_lr.coefficients.toArray
val coeffs = if (pipeline_model_lr.getFitIntercept) {
  Array(pipeline_model_lr.intercept) ++ lrModelCoeffs
} else {
  lrModelCoeffs
}
println("Feature\tCoefficient")
featureNames.zip(coeffs).foreach { case (feature, coeff) =>
  println(s"$feature\t$coeff")
}
```

## Model Serialization and Deserialization for Deployment 

In order to deploy the trained model to the production, we first serialize the _`final_pipeline_gbt`_ object into a single JSON file using MLeap. Serialization using MLeap is simple and straightforward and it supports serializing model to a directory or a zip file in the format of either JSON or Protobuf.
```scala
import ml.combust.bundle.BundleFile
import ml.combust.bundle.serializer.SerializationFormat
import ml.combust.mleap.spark.SparkSupport._
import resource._

for(bundle <- managed(BundleFile("jar:file:/chao/mymodel.zip"))) {
  final_pipeline_gbt.writeBundle.format(SerializationFormat.Json).save(bundle)
}
```
The model deployment is implemented as a service via the REST API. Simply take the saved JSON file and deserialize it inside a Scala web framework. This articule <https://auth0.com/blog/build-and-secure-a-scala-play-framework-api/> demonstrates how to build an API using Scala _**Play**_ framework.
```scala
import ml.combust.bundle.BundleFile
import ml.combust.mleap.runtime.MleapSupport._
import resource._

val zipBundleM = (for(bundle <- managed(BundleFile("jar:file:/chao/mymodel.zip"))) yield {
  bundle.loadMleapBundle().get
}).opt.get

val loaded_pipeline_gbt = zipBundleM.root
```
## Conclusion
This post elaborates on the process of building a machine learning model pipeline in Spark, with the code snippets providing all the details for the implementation from data import, preprocessing, feature engineering, model tuning and training, to its deployment. Following this protocol enables myself to build a predictive model in production and serve our business. Hopefully this can be also helpful as a tutorial for people who are struggling with the Spark machine learning process.
