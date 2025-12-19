
package com.example.datacollection

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color as ComposeColor
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.app.ActivityCompat
import androidx.core.content.FileProvider
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import java.io.File
import java.io.FileWriter

class MainActivity : ComponentActivity(), SensorEventListener, LocationListener {

    private lateinit var sensorManager: SensorManager
    private lateinit var locationManager: LocationManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private val viewModel: SensorViewModel by viewModels()

    // Buffers for sensor data
    private val accelerometerBuffer = mutableListOf<FloatArray>()
    private val gyroscopeBuffer = mutableListOf<FloatArray>()
    private var lastChartUpdateTime: Long = 0

    private val locationPermissionRequest = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions[Manifest.permission.ACCESS_FINE_LOCATION] == true) {
            startLocationUpdates()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        setContent {
            SensorDataScreen(viewModel, onToggleRecording = {
                viewModel.toggleRecording()
                registerSensorListeners(viewModel.isRecording.value)
                if (!viewModel.isRecording.value) {
                    saveAndShareCsv()
                }
            })
        }

        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            locationPermissionRequest.launch(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION))
        } else {
            startLocationUpdates()
        }
    }

    private fun registerSensorListeners(isRecording: Boolean) {
        sensorManager.unregisterListener(this)
        val sensorDelay = if (isRecording) SensorManager.SENSOR_DELAY_FASTEST else SensorManager.SENSOR_DELAY_GAME
        accelerometer?.also { accel ->
            sensorManager.registerListener(this, accel, sensorDelay)
        }
        gyroscope?.also { gyro ->
            sensorManager.registerListener(this, gyro, sensorDelay)
        }
    }

    override fun onResume() {
        super.onResume()
        registerSensorListeners(viewModel.isRecording.value)
        startLocationUpdates()
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
        locationManager.removeUpdates(this)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Not used
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                synchronized(accelerometerBuffer) {
                    accelerometerBuffer.add(event.values.clone())
                }
            }
            Sensor.TYPE_GYROSCOPE -> {
                synchronized(gyroscopeBuffer) {
                    gyroscopeBuffer.add(event.values.clone())
                }
            }
        }

        val currentTime = System.currentTimeMillis()
        if (currentTime - lastChartUpdateTime < 100) {
            return
        }
        lastChartUpdateTime = currentTime

        var avgAccel: FloatArray? = null
        var avgGyro: FloatArray? = null

        synchronized(accelerometerBuffer) {
            if (accelerometerBuffer.isNotEmpty()) {
                val avgX = accelerometerBuffer.map { it[0] }.average().toFloat()
                val avgY = accelerometerBuffer.map { it[1] }.average().toFloat()
                val avgZ = accelerometerBuffer.map { it[2] }.average().toFloat()
                avgAccel = floatArrayOf(avgX, avgY, avgZ)
                accelerometerBuffer.clear()
            }
        }

        synchronized(gyroscopeBuffer) {
            if (gyroscopeBuffer.isNotEmpty()) {
                val avgX = gyroscopeBuffer.map { it[0] }.average().toFloat()
                val avgY = gyroscopeBuffer.map { it[1] }.average().toFloat()
                val avgZ = gyroscopeBuffer.map { it[2] }.average().toFloat()
                avgGyro = floatArrayOf(avgX, avgY, avgZ)
                gyroscopeBuffer.clear()
            }
        }

        avgAccel?.let { viewModel.setAccelerometerData(it[0], it[1], it[2], event.timestamp) }
        avgGyro?.let { viewModel.setGyroscopeData(it[0], it[1], it[2], event.timestamp) }

        if (viewModel.isRecording.value) {
            val roadCondition = if(viewModel.isGoodRoad.value) "Good" else "Bad"
            val ax = avgAccel?.get(0) ?: viewModel.accelerometerData.value.x
            val ay = avgAccel?.get(1) ?: viewModel.accelerometerData.value.y
            val az = avgAccel?.get(2) ?: viewModel.accelerometerData.value.z
            val wx = avgGyro?.get(0) ?: viewModel.gyroscopeData.value.x
            val wy = avgGyro?.get(1) ?: viewModel.gyroscopeData.value.y
            val wz = avgGyro?.get(2) ?: viewModel.gyroscopeData.value.z
            
            viewModel.recordedData.add("${System.currentTimeMillis()},$ax,$ay,$az,$wx,$wy,$wz,${viewModel.gpsData.value.latitude},${viewModel.gpsData.value.longitude},${viewModel.gpsData.value.speed},$roadCondition")
        }
    }

    override fun onLocationChanged(location: Location) {
        viewModel.setGpsData(location.latitude, location.longitude, location.speed, location.time)
    }

    private fun startLocationUpdates() {
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            locationManager.allProviders.forEach { provider ->
                locationManager.requestLocationUpdates(provider, 500, 0f, this)
            }
        }
    }

    private fun saveAndShareCsv() {
        val fileDir = getExternalFilesDir("recordings")
        if (fileDir != null && !fileDir.exists()) {
            fileDir.mkdirs()
        }

        val csvFile = File(fileDir, "sensor_data.csv")

        try {
            FileWriter(csvFile).use { writer ->
                writer.append("timestamp,ax,ay,az,wx,wy,wz,latitude,longitude,speed,roadCondition\n")
                viewModel.recordedData.forEach { line ->
                    writer.append(line)
                    writer.append("\n")
                }
            }
            viewModel.recordedData.clear()

            val fileUri = FileProvider.getUriForFile(this, "${applicationContext.packageName}.provider", csvFile)

            val shareIntent = Intent().apply {
                action = Intent.ACTION_SEND
                putExtra(Intent.EXTRA_STREAM, fileUri)
                type = "text/csv"
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(shareIntent, "Share CSV"))

        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}

@Composable
fun SensorDataScreen(viewModel: SensorViewModel, onToggleRecording: () -> Unit) {
    val isGoodRoad by viewModel.isGoodRoad
    val isRecording by viewModel.isRecording

    Box(modifier = Modifier.fillMaxSize().background(ComposeColor.Black)) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
        ) {
            Spacer(modifier = Modifier.height(64.dp))
            val accelerometerData by viewModel.accelerometerData
            val gyroscopeData by viewModel.gyroscopeData
            val gpsData by viewModel.gpsData
            val totalAcceleration by viewModel.totalAcceleration
            val totalAngularVelocity by viewModel.totalAngularVelocity
            SensorCard(
                sensorName = "Gyroscope",
                sensorData = gyroscopeData,
                totalValue = totalAngularVelocity,
                xLabel = "ωx",
                yLabel = "ωy",
                zLabel = "ωz",
                unit = "rad/s"
            )
            SensorCard(
                sensorName = "Accelerometer",
                sensorData = accelerometerData,
                totalValue = totalAcceleration,
                xLabel = "ax",
                yLabel = "ay",
                zLabel = "az",
                unit = "m/s²"
            )
            GpsCard(gpsData = gpsData)
            Spacer(modifier = Modifier.height(64.dp))
        }

        Row(
            modifier = Modifier.align(Alignment.TopCenter).padding(top = 16.dp),
            horizontalArrangement = Arrangement.Center
        ){
            Button(
                onClick = { viewModel.toggleRoadCondition() },
                colors = ButtonDefaults.buttonColors(containerColor = if (isGoodRoad) ComposeColor.Green else ComposeColor.Red)
            ) {
                Text(text = if (isGoodRoad) "Good Road" else "Bad Road")
            }
        }

        Button(
            onClick = onToggleRecording,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(16.dp)
                .fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = if (isRecording) ComposeColor.Red else ComposeColor.Green)
        ) {
            Text(text = if (isRecording) "Stop Recording" else "Start Recording")
        }
    }
}

@Composable
fun SensorCard(
    sensorName: String,
    sensorData: SensorData,
    totalValue: Float,
    xLabel: String,
    yLabel: String,
    zLabel: String,
    unit: String
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = ComposeColor(0xFF2E2E2E))
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = sensorName,
                color = ComposeColor.White,
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(16.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                SensorValue(label = xLabel, value = sensorData.x, color = ComposeColor(0xFFFF073A))
                SensorValue(label = yLabel, value = sensorData.y, color = ComposeColor(0xFF39FF14))
                SensorValue(label = zLabel, value = sensorData.z, color = ComposeColor(0xFF04D9FF))
            }
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = "Total = %.2f $unit".format(totalValue),
                color = ComposeColor.White,
                fontSize = 18.sp
            )
            Spacer(modifier = Modifier.height(16.dp))
            LineChart(sensorData = sensorData, description = sensorName)
        }
    }
}

@Composable
fun GpsCard(gpsData: GpsData) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = ComposeColor(0xFF2E2E2E))
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "GPS",
                color = ComposeColor.White,
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(16.dp))
            Text(text = "Latitude: ${gpsData.latitude}", color = ComposeColor.White)
            Text(text = "Longitude: ${gpsData.longitude}", color = ComposeColor.White)
            Text(text = "Speed: ${gpsData.speed} m/s", color = ComposeColor.White)
            Spacer(modifier = Modifier.height(16.dp))
            SpeedChart(gpsData = gpsData)
        }
    }
}

@Composable
fun SensorValue(label: String, value: Float, color: ComposeColor) {
    Box(
        modifier = Modifier
            .background(color, RoundedCornerShape(8.dp))
            .padding(8.dp)
    ) {
        Text(text = "$label: %.2f".format(value), color = ComposeColor.White)
    }
}

@Composable
fun LineChart(sensorData: SensorData, description: String) {
    AndroidView(
        modifier = Modifier
            .fillMaxWidth()
            .height(200.dp),
        factory = { context ->
            val chart = LineChart(context)
            chart.apply {
                // Performance Optimizations
                this.description.text = description
                this.description.textColor = Color.WHITE
                setDrawGridBackground(false)
                setDrawBorders(false)
                legend.isEnabled = false

                // General Chart Settings
                isDragEnabled = true
                setScaleEnabled(true)
                setPinchZoom(true)

                // Y-Axis (Left) - Enable Auto-Scaling
                axisLeft.setDrawGridLines(false)
                axisLeft.textColor = Color.WHITE

                // Y-Axis (Right)
                axisRight.isEnabled = false

                // X-Axis
                xAxis.setDrawGridLines(false)
                xAxis.setDrawLabels(false)
                xAxis.textColor = Color.WHITE

                // Data Setup
                val dataSetX = LineDataSet(null, "X").apply {
                    mode = LineDataSet.Mode.LINEAR
                    setDrawCircles(false)
                    setDrawValues(false)
                    color = Color.parseColor("#FF073A") // Neon Red
                    valueTextColor = Color.WHITE
                }
                val dataSetY = LineDataSet(null, "Y").apply {
                    mode = LineDataSet.Mode.LINEAR
                    setDrawCircles(false)
                    setDrawValues(false)
                    color = Color.parseColor("#39FF14") // Neon Green
                    valueTextColor = Color.WHITE
                }
                val dataSetZ = LineDataSet(null, "Z").apply {
                    mode = LineDataSet.Mode.LINEAR
                    setDrawCircles(false)
                    setDrawValues(false)
                    color = Color.parseColor("#04D9FF") // Neon Blue
                    valueTextColor = Color.WHITE
                }
                this.data = LineData(dataSetX, dataSetY, dataSetZ)

                // Use the tag to store the persistent x-value counter
                this.tag = 0f
            }
            chart
        },
        update = { chart ->
            if (sensorData.timestamp == 0L) return@AndroidView // Don't draw initial empty data

            val data = chart.data
            if (data != null) {
                // Retrieve and increment the persistent x-value from the tag
                var xValue = chart.tag as? Float ?: 0f
                chart.tag = xValue + 1f

                data.addEntry(Entry(xValue, sensorData.x), 0)
                data.addEntry(Entry(xValue, sensorData.y), 1)
                data.addEntry(Entry(xValue, sensorData.z), 2)

                // Limit the number of entries
                val setX = data.getDataSetByIndex(0)
                if (setX.entryCount > 300) {
                    setX.removeFirst()
                    data.getDataSetByIndex(1).removeFirst()
                    data.getDataSetByIndex(2).removeFirst()
                }

                // Let the chart know when data has changed
                chart.notifyDataSetChanged()

                // Limit the number of visible entries
                chart.setVisibleXRangeMaximum(150f)

                // Move to the latest entry
                chart.moveViewToX(xValue)
            }
        }
    )
}

@Composable
fun SpeedChart(gpsData: GpsData) {
    AndroidView(
        modifier = Modifier
            .fillMaxWidth()
            .height(200.dp),
        factory = { context ->
            val chart = LineChart(context)
            chart.apply {
                this.description.text = "Speed"
                this.description.textColor = Color.WHITE
                setDrawGridBackground(false)
                setDrawBorders(false)
                legend.isEnabled = false

                isDragEnabled = true
                setScaleEnabled(true)
                setPinchZoom(true)

                axisLeft.setDrawGridLines(false)
                axisLeft.textColor = Color.WHITE

                axisRight.isEnabled = false

                xAxis.setDrawGridLines(false)
                xAxis.setDrawLabels(false)
                xAxis.textColor = Color.WHITE

                val dataSet = LineDataSet(null, "Speed").apply {
                    mode = LineDataSet.Mode.LINEAR
                    setDrawCircles(false)
                    setDrawValues(false)
                    color = Color.parseColor("#FFD700") // Gold
                    valueTextColor = Color.WHITE
                }
                this.data = LineData(dataSet)
                this.tag = 0f
            }
            chart
        },
        update = { chart ->
            if (gpsData.timestamp == 0L) return@AndroidView

            val data = chart.data
            if (data != null) {
                var xValue = chart.tag as? Float ?: 0f
                chart.tag = xValue + 1f

                data.addEntry(Entry(xValue, gpsData.speed), 0)

                val set = data.getDataSetByIndex(0)
                if (set.entryCount > 300) {
                    set.removeFirst()
                }

                chart.notifyDataSetChanged()
                chart.setVisibleXRangeMaximum(150f)
                chart.moveViewToX(xValue)
            }
        }
    )
}
