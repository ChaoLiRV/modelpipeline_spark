
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
