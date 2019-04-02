# Create and Deploy a Machine Learning Model Pipeline in Spark
As data scientists at Red Ventures, we define ourselves as [type _'B'_ data scientists]( <https://www.dezyre.com/article/type-a-data-scientist-vs-type-b-data-scientist/194>), which differs from the role of business intelligence or data analysts. Besides providing business insights to drive actionable items, we are also dedicated to building models from end to end and deploying them as a service for business usage, both internally and externally. This post describes the general process of building a classification model pipeline in **Spark** and touches upon its deployment via a REST API with code snippets that provide all the details for implementation from data import, preprocessing, feature engineering, model tuning and training to its deployment.

_The Spark code in this post is written in **Scala** and run on the **Databricks** platform_

## Data Import
The first thing we need to do is import the data for training our model. Our data, in this example, is stored in the Snowflake warehouse in the cloud. To enable our SQL queries to read data from the Snowflake database, we'll use the [databricks-snowflake connector](https://docs.databricks.com/spark/latest/data-sources/snowflake.html). When we have a connection, we'll use a Scala function to query and read from the Snowflake database in the cloud.

The following `readDataFromSnowflake` function does the following:
- Configures the connection with Snowflake
- Executes a query to select all data fields from the Snowflake table `myTable` under the schema `mySchema` for a particular date range, specified by the function arguments
- Returns the entire dataset as a Spark DataFrame

```scala
import net.snowflake.spark.snowflake.SnowflakeConnectorUtils
import net.snowflake.spark.snowflake.Utils.SNOWFLAKE_SOURCE_NAME
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._

def readDataFromSnowflake(dateStart: String, dateStop: String) : DataFrame = {
  // dateStart:String e.g. "2017-12-01"
  // dateStop:String e.g. "2018-06-01"

  // Snowflake configuration
  // Note: dbutils are utilities available on the Databricks platform, including functionality used below to securely store and use credentials without making them visible (more info here: https://docs.databricks.com/user-guide/dev-tools/dbutils.html)
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

To automate the model update process, the date range is extracted from the system datetime every time the job is running. The code below implements the function to obtain the training data from a six month (180 days) look-back window. A snapshot of this dataset suggests the features are comprised of various types, such as categorical variables _Var1, Var2, Var3_, binary variable _Var4_, and numeric variable types _Var5_ and _Var6_. Here _label_ is the classification target that the model learns to predict.

```scala
import java.util.{Calendar, Date}
import java.text.SimpleDateFormat

val formatter = new SimpleDateFormat("YYYY-MM-dd")
val calendar = Calendar.getInstance()
val dateStop: String = formatter.format(calendar.getTime()) // get current date
calendar.add(Calendar.DAY_OF_YEAR, -180) // look back over the past 180 days
val dateStart: String = formatter.format(calendar.getTime())

val df_all: DataFrame = readDataFromSnowflake(dateStart, dateStop)
display(df_all.limit(5))
```

| ID   | Var1       | Var2     |Var3      | Var4 | Var5      | Var6 | label |
|------|------------|----------|----------|------|-----------|------|-------|
| 0001 | SmartPhone | NULL     | Off-Peak | 1    | 484500.47 | 16   | 0     |
| 0002 | Mobile     | Firefox  | Peak     | 0    | 513500.32 | NaN  | 1     |
| 0003 | Desktop    | Chrome   | Off-Peak | 0    | 441000.45 | 25   | 0     |
| 0004 | Desktop    | Safari   | Weekend  | 0    | 840000.4  | 2    | 0     |
| 0005 | Desktop    | Chrome   | Peak     | 1    | NaN       | 22   | 1     |

_Var1-3_ represent the device type, browser type, and time range respectively. To make these variable names more descriptive, you can rename the dataframe columns using Spark Dataframe's method `withColumnRenamed()`. In this exercise, I found a really useful function [foldLeft()](http://allaboutscala.com/tutorials/chapter-8-beginner-tutorial-using-scala-collection-functions/scala-foldleft-example/) that makes this process more scalable in the sense that you need not write 100 statements for 100 variable name changes. All that's needed is to make the _old name_ and _new name_ a key-value pair and store it in the scala _Map_ value. No other code change is required. Another benefit of the `foldLeft()` function is that you don't have to loop through an overwrite a `var` (anything which is mutable is riskier), but can loop through a `val` and output a `val`.

```scala
/* rename column names */
// df_all = df_all.withColumnRenamed("Var1", "Device")
// df_all = df_all.withColumnRenamed("Var2", "Browser")
// df_all = df_all.withColumnRenamed("Var3", "TimeRange")

/* rename column names using foldLeft */
val columnNameMap = Map("Var1"->"Device",
                        "Var2"->"Browser",
                        "Var3"->"TimeRange")
val df_all_renamed = columnNameMap.keys.foldLeft(df_all){
  (tmpDF, colName) =>
  tmpDF.withColumnRenamed(colName, columnNameMap(colName) )
}

display(df_all_renamed.limit(5))
```

| ID   | Device     | Browser       | TimeRange| Var4 | Var5      | Var6 | label |
|------|------------|---------------|----------|------|-----------|------|-------|
| 0001 | SmartPhone | NULL          | Off-Peak | 1    | 484500.47 | 16   | 0     |
| 0002 | Mobile     | Firefox       | Peak     | 0    | 513500.32 | NaN  | 1     |
| 0003 | Desktop    | Chrome        | Off-Peak | 0    | 441000.45 | 25   | 0     |
| 0004 | Desktop    | Safari        | Weekend  | 0    | 840000.4  | 2    | 0     |
| 0005 | Desktop    | Chrome        | Peak     | 1    | NaN       | 22   | 1     |


Because the subject of this article is building a model pipeline for deployment, the data exploration section is done offline and not described here. Next, we'll look at pipeline creation for feature engineering and model training.

```scala
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.attribute._

// Split training- testing dataset
val splitSeed = 1234
val Array(trainingData, testingData) = df_all_renamed.randomSplit(Array(0.8, 0.2), splitSeed)
```

## Feature Transformer Pipeline

#### Numeric Variables
For a model running in production, it is always a good habit to set a defensive layer to handle any anomalies gracefully. In this example, we set an **Imputer** transformer in the pipeline to handle the missing values for numeric variables _Var5_ and _Var6_.

This process generates two additional columns _Var5Impute_ and _Var6Impute_ that replace the _NaN_ value in original column _Var5_ and _Var6_ by their respective median (see the example outcome below).

**Note**: The choice of the transformer is, to some extent, limited to the availability in [**MLeap**](http://mleap-docs.combust.ml/core-concepts/transformers/support.html). MLeap is the tool that we use to serialize the Spark model pipelines, and we'll touch on that later in this post. If the transformer function that you need doesn't exist on their list, follow [the procedure here](http://mleap-docs.combust.ml/mleap-runtime/custom-transformer.html) to create the custom transformer.

```scala
import org.apache.spark.ml.mleap.feature.Imputer
import org.apache.spark.ml.feature.ImputerModel

// Configure the imputer for each numeric column
val var5Imputer: ImputerModel = new Imputer()
  .setInputCol("Var5")
  .setOutputCol("Var5Impute")
  .setStrategy("median")
val var6Imputer: ImputerModel = new Imputer()
  .setInputCol("Var6")
  .setOutputCol("Var6Impute")
  .setStrategy("median")

// For a look at how the imputation works, let's fit the imputers and examine the transformed output
val var5ImputerModel: ImputerModel = var5Imputer.fit(trainingData)
val var6ImputerModel: ImputerModel = var6Imputer.fit(trainingData)
val listOfImputers: List[ImputerModel] = List(var5ImputerModel, var6ImputerModel)

// Loop through each imputer and transform the original training data
val trainingData_imputed = listOfImputers.foldLeft(trainingData) { (prev: DataFrame, curr: ImputerModel) =>
  curr.transform(prev)
}

display(trainingData_imputed)
```

| ID        | Device     | Browser       | TimeRange | Var4 | Var5      | Var6 | label | Var5Impute | Var6Impute |
|-----------|------------|---------------|-----------|------|-----------|------|-------|------------|------------|
| 136581531 | SmartPhone | NULL          | Off-Peak  | 1    | 484500.47 | 16   | 0     | 484500.47  | 16         |
| 138079502 | Mobile     | Firefox       | Peak      | 0    | 513500.32 | NaN  | 1     | 513500.32  | 16         |
| 136280501 | Desktop    | Chrome        | Off-Peak  | 0    | 441000.45 | 25   | 0     | 441000.45  | 25         |
| 136608744 | Desktop    | Safari        | Weekend   | 0    | 840000.4  | 2    | 0     | 840000.4   | 2          |
| 136576205 | Desktop    | Chrome        | Peak      | 1    | NaN       | 22   | 1     | 275500.5   | 22         |

Next, we perform numeric operations on these variables, such as dividing one feature value by the other, taking the logarithmic transform of the value, and scale normalization (Min-Max or Z-score).

The columns created by these operations are concatenated to the table as shown in the following example:

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
                .setOutputCol("Var5_to_Var6")
// unary operation (log transformation) on a single feature
val logTransformer = new MathUnary(uid = "ltvlog", model = MathUnaryModel(Log)) //
                .setInputCol("Var5Impute")
                .setOutputCol("Var5_Log")

val assemblerNum = new VectorAssembler().setInputCols(Array("Var5_to_Var6","Var5_Log"))
                                        .setOutputCol("numeric_features_vec")
val scaler = new feature.StandardScaler()
                .setInputCol("numeric_features_vec")
                .setOutputCol("scaled_numeric_features")
                .setWithStd(true)
                .setWithMean(true)
```

| ID        | Device     | Browser       | TimeRange | Var4 | Var5      | Var6 | label | Var5Impute | Var6Impute | Var5_to_Var6 | Var5_Log    | numeric_features_vec       | scaled_numeric_features |
|-----------|------------|---------------|-----------|------|-----------|------|-------|------------|------------|--------------|-------------|----------------------------|-------------------------|
| 136581531 | SmartPhone | NULL          | Off-Peak  | 1    | 484500.47 | 16   | 0     | 484500.47  | 16         | 30281.27938  | 5.685294203 | [30281.27938, 5.685294203] | [-0.1236, 0.9264]       |
| 138079502 | Mobile     | Firefox       | Peak      | 0    | 513500.32 | NaN  | 1     | 513500.32  | 16         | 32093.77     | 5.710540719 | [32093.77, 5.710540719]    | [-0.1156, 1.0527]       |
| 136280501 | Desktop    | Chrome        | Off-Peak  | 0    | 441000.45 | 25   | 0     | 441000.45  | 25         | 17640.018    | 5.644439033 | [17640.018, 5.644439033]   | [-0.1797, 0.7221]       |
| 136608744 | Desktop    | Safari        | Weekend   | 0    | 840000.4  | 2    | 0     | 840000.4   | 2          | 420000.2     | 5.924279493 | [420000.2, 5.924279493]    | [1.6060, 2.1213]        |
| 136576205 | Desktop    | Chrome        | Peak      | 1    | NaN       | 22   | 1     | 275500.5   | 22         | 12522.75     | 5.440122391 | [12522.75, 5.440122391]    | [-0.2024, -0.2993]      |

#### Categorical Variables
The same goes for categorical variables. At the beginning, we'll set an imputation stage for handling missing values. MLeap doesn't provide this transformer function, as you can't find it on [this list](http://mleap-docs.combust.ml/core-concepts/transformers/support.html). Therefore, we've written a custom transformer, _**StringImputer**_, by following the aforementioned MLeap document. This transformer imputes missing data with a String value of your choice.

For categorical variables, there are instances when we might want to bucket two Strings that represent similar concepts, such as _"Mobile"_ and _"SmartPhone"_. In this situation, the _**StringMapper**_ transformer is employed to achieve this. Note that I utilize a custom transformer in the code instead of the MLeap built-in _StringMap_, because _StringMap_ doesn't allow the default value to be set in the map.

Next, we'll utilize _**StringIndexer**_ as an additional defensive layer for handling unseen values during training. The categorical values are mapped to numeric index based off of the frequency. For example, the dataset contains more _Desktop_ than _SmartPhone_ and, thus, their corresponding string indices are 0.0 and 1.0. An unseen device value like _Console_ will be mapped to 2.0.

See the following table with the columns _Device_Impute_, _Device_Map_ and _Device_Index_. For the best of comparison, the numeric variable columns are not shown here.

```scala
import com.redventures.custom.core.transformer.StringMapperModel
import org.apache.spark.ml.custom.transformer.StringMapper
import org.apache.spark.ml.custom.transformer.StringImputer

/* Device Transformers*/
val deviceImputer = new StringImputer(uid = "device_imp", model = StringImputerModel("OtherDevices"))
                .setInputCol("Device")
                .setOutputCol("Device_Impute")

val deviceMapper = new StringMapper(uid = "device_map", model = StringMapperModel(
                Map("Mobile"->"SmartPhone","SmartPhone"->"SmartPhone","Desktop"->"Desktop"), "OtherDevices") ) //
                .setInputCol("Device_Impute")
                .setOutputCol("Device_Map")

val deviceIndexer = new feature.StringIndexer()
                .setInputCol("Device_Map")
                .setOutputCol("Device_Index")
                .setHandleInvalid("keep")
```
| ID        | Device     | Browser | TimeRange | Device_Impute | Device_Map | Device_Index |
|-----------|------------|---------|-----------|---------------|------------|--------------|
| 136581531 | SmartPhone | NULL    | Off-Peak  | SmartPhone    | SmartPhone | 1.0          |
| 138079502 | Mobile     | Firefox | Peak      | Mobile        | SmartPhone | 1.0          |
| 136280501 | Desktop    | Chrome  | Off-Peak  | Desktop       | Desktop    | 0.0          |
| 136608744 | Desktop    | Safari  | Weekend   | Desktop       | Desktop    | 0.0          |
| 136576205 | Desktop    | Chrome  | Peak      | Desktop       | Desktop    | 0.0          |

Repeat the procedures to apply transformers to other categorical variables _TimeRange_ and _Browser_. And then set the _one-hot encoding_ stage for all processed categorical variables. The output values by one-hot-encoding are represented in a sparse format. For example, _(2, [1], [1.0])_ in _Device_OHE_ indicates a vector of length of 2 with 1.0 at position 1 and 0 elsewhere.

```scala
/* TimeRange */
val timeImputer = new StringImputer(uid = "time_imp", model = StringImputerModel("Peak"))
                .setInputCol("TimeRange")
                .setOutputCol("Time_Impute")
val timeIndexer = new StringIndexer()
                .setInputCol("Time_Impute")
                .setOutputCol("Time_Index")
                .setHandleInvalid("keep")
/* Browser */
val browserImputer = new StringImputer(uid = "browser_imp", model = StringImputerModel("OtherBrowser"))
                .setInputCol("Browser")
                .setOutputCol("Browser_Impute")
val browserIndexer = new StringIndexer()
                .setInputCol("Browser_Impute")
                .setOutputCol("Browser_Index")
                .setHandleInvalid("keep")

/* One hot encodings */
val ohes = new OneHotEncoderEstimator()
              .setInputCols(Array("Device_Index","Time_Index","Browser_Index"))
              .setOutputCols(Array("Device_OHE","Time_OHE","Browser_OHE"))
              .setDropLast(true)
```
| ID        | Device     | Browser | TimeRange | Device_Impute | Device_Map | Device_Index | Time_Impute | Time_Index | Browser_Impute | Browser_Index | Device_OHE      | Time_OHE        | Browser_OHE     |
|-----------|------------|---------|-----------|---------------|------------|--------------|-------------|------------|----------------|---------------|-----------------|-----------------|-----------------|
| 136581531 | SmartPhone | NULL    | Off-Peak  | SmartPhone    | SmartPhone | 1.0          | Off-Peak    | 2.0        | OtherBrowser   | 3.0           | (2, [1], [1.0]) | (3, [2], [1.0]) | (4, [3], [1.0]) |
| 138079502 | Mobile     | Firefox | Peak      | Mobile        | SmartPhone | 1.0          | Peak        | 0.0        | Firefox        | 1.0           | (2, [1], [1.0]) | (3, [0], [1.0]) | (4, [1], [1.0]) |
| 136280501 | Desktop    | Chrome  | Off-Peak  | Desktop       | Desktop    | 0.0          | Off-Peak    | 2.0        | Chrome         | 0.0           | (2, [0], [1.0]) | (3, [2], [1.0]) | (4, [0], [1.0]) |
| 136608744 | Desktop    | Safari  | Weekend   | Desktop       | Desktop    | 0.0          | Weekend     | 1.0        | Safari         | 2.0           | (2, [0], [1.0]) | (3, [1], [1.0]) | (4, [2], [1.0]) |
| 136576205 | Desktop    | Chrome  | Peak      | Desktop       | Desktop    | 0.0          | Peak        | 0.0        | Chrome         | 0.0           | (2, [0], [1.0]) | (3, [0], [1.0]) | (4, [0], [1.0]) |

The classification model in the **spark.ml** package is an estimator, which requires all feature data to be assembled as a vector for each record in the column _`"feature"`_. This is done by the transformer `VectorAssembler()` and, again, the output values are represented in a sparse format as shown in the following table.

Lastly, we stack all the aforementioned transformers in the sequence to a pipeline object. As a consequence, every single record of both the training and testing set is guaranteed to go through the same feature engineering process without incurring exception by anomaly value.

```scala
/* assemble all processed features into a single vector */
val featureCols = Array("Device_OHE", "Time_OHE", "Browser_OHE", "Var4", "scaled_numeric_features")
val assembler_all = new feature.VectorAssembler().setInputCols(featureCols).setOutputCol("features")

// set the pipeline to include all the feature engineering stages
val feature_stages = new Pipeline()
                .setStages( Array(var5Imputer, var6Imputer, // numeric variable imputer
                                  divider, logTramsformer, assembler_num, scaler,  //numeric variable processing
                                  deviceImputer, deviceMapper, deviceIndexer, // Device type processing
                                  timeImputer, timeIndexer, // Time range
                                  browserImputer, browserIndexer, // Browser
                                  ohes, // one hot encoding
                                  assembler_all))

val df_transformed = feature_stages.transform(trainingData)
display(df_transformed)
```

| ID        | Device     | Browser | TimeRange | Var4 | Var5      | Var6 | label | Var5Impute | Var6Impute | Var5_to_Var6 | Var5_Log    | numeric_features_vec       | scaled_numeric_features | Device_Impute | Device_Map | Device_Index | Time_Impute | Time_Index | Browser_Impute | Browser_Index | Device_OHE      | Time_OHE        | Browser_OHE     | features                                           |
|-----------|------------|---------|-----------|------|-----------|------|-------|------------|------------|--------------|-------------|----------------------------|-------------------------|---------------|------------|--------------|-------------|------------|----------------|---------------|-----------------|-----------------|-----------------|----------------------------------------------------|
| 136581531 | SmartPhone | NULL    | Off-Peak  | 1    | 484500.47 | 16   | 0     | 484500.47  | 16         | 30281.27938  | 5.685294203 | [30281.27938, 5.685294203] | [-0.1236, 0.9264]       | SmartPhone    | SmartPhone | 1.0          | Off-Peak    | 2.0        | OtherBrowser   | 3.0           | (2, [1], [1.0]) | (3, [2], [1.0]) | (4, [3], [1.0]) | [(12, [1,4,8,9,10,11], [1,1,1,1,-0.1236,0.9264 ])] |
| 138079502 | Mobile     | Firefox | Peak      | 0    | 513500.32 | NaN  | 1     | 513500.32  | 16         | 32093.77     | 5.710540719 | [32093.77, 5.710540719]    | [-0.1156, 1.0527]       | Mobile        | SmartPhone | 1.0          | Peak        | 0.0        | Firefox        | 1.0           | (2, [1], [1.0]) | (3, [0], [1.0]) | (4, [1], [1.0]) | [(12, [1,2,6,10,11], [1,1,1,-0.1156,1.0527])]      |
| 136280501 | Desktop    | Chrome  | Off-Peak  | 0    | 441000.45 | 25   | 0     | 441000.45  | 25         | 17640.018    | 5.644439033 | [17640.018, 5.644439033]   | [-0.1797, 0.7221]       | Desktop       | Desktop    | 0.0          | Off-Peak    | 2.0        | Chrome         | 0.0           | (2, [0], [1.0]) | (3, [2], [1.0]) | (4, [0], [1.0]) | [(12, [0,4,5,10,11], [1,1,1,-0.1797,0.7221])]      |
| 136608744 | Desktop    | Safari  | Weekend   | 0    | 840000.4  | 2    | 0     | 840000.4   | 2          | 420000.2     | 5.924279493 | [420000.2, 5.924279493]    | [1.6060, 2.1213]        | Desktop       | Desktop    | 0.0          | Weekend     | 1.0        | Safari         | 2.0           | (2, [0], [1.0]) | (3, [1], [1.0]) | (4, [2], [1.0]) | [(12, [0,3,7,10,11], [1,1,1,1.6060,2.1213])]       |
| 136576205 | Desktop    | Chrome  | Peak      | 1    | NaN       | 22   | 1     | 275500.5   | 22         | 12522.75     | 5.440122391 | [12522.75, 5.440122391]    | [-0.2024, -0.2993]      | Desktop       | Desktop    | 0.0          | Peak        | 0.0        | Chrome         | 0.0           | (2, [0], [1.0]) | (3, [0], [1.0]) | (4, [0], [1.0]) | [(12, [0,2,5,9,10,11], [1,1,1,1,-0.2024,-0.2993])] |

## Model Estimator
In this example, the gradient boosting classifier _GBTClassifier_ is chosen to do the predictive task as it achieves state-of-the-art performance. The _CrossValidator_ is employed for the model tuning and the _Parallelizable_ version of cross validator can speed up the computation. Once the model parameters are determined by the tuning, the model estimator is appended to the pipeline as the last stage. The pipeline object's _`fit`_ method executes the entire workflow, including both the feature engineering and model training process on the dataset.

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
Because this is a binary classification task, the area under the _ROC_ curve is calculated to measure the model performance. In addition, the gradient boosting models are also evaluated against the vanilla version model's logistic regression.
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
Logistic regression is considered a white box algorithm. The code below extracts the coefficients for the model.
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
To deploy the trained model to the production, we first serialize the _`final_pipeline_gbt`_ object into a single JSON file using MLeap. Serialization using MLeap is simple and straightforward, and it supports serializing models to a directory or a .zip file in the format of either JSON or Protobuf.
```scala
import ml.combust.bundle.BundleFile
import ml.combust.bundle.serializer.SerializationFormat
import ml.combust.mleap.spark.SparkSupport._
import resource._

for(bundle <- managed(BundleFile("jar:file:/chao/mymodel.zip"))) {
  final_pipeline_gbt.writeBundle.format(SerializationFormat.Json).save(bundle)
}
```

The model deployment is implemented as a service via the REST API. Simply take the saved JSON file and deserialize it inside a Scala web framework. [This article](https://auth0.com/blog/build-and-secure-a-scala-play-framework-api/) demonstrates how to build an API using Scala _**Play**_ framework.
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
This post elaborates on the process of building a machine learning model pipeline in Spark, with the code snippets providing all the details for the implementation from data import, preprocessing, feature engineering, model tuning and training to its deployment. This protocol enables me to build a predictive model in production and serve our business. Hopefully this can also be a helpful tutorial for people who are new to the Spark machine learning process.

_Interested in solving complex problems and building products that drive value? Come check us out at RedVentures.com._