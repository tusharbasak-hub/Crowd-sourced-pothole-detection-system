package com.example.datacollection

import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import kotlin.math.sqrt

data class SensorData(val x: Float, val y: Float, val z: Float, val timestamp: Long)
data class GpsData(val latitude: Double, val longitude: Double, val speed: Float, val timestamp: Long)

class SensorViewModel : ViewModel() {
    private val _accelerometerData = mutableStateOf(SensorData(0f, 0f, 0f, 0L))
    val accelerometerData: State<SensorData> = _accelerometerData

    private val _gyroscopeData = mutableStateOf(SensorData(0f, 0f, 0f, 0L))
    val gyroscopeData: State<SensorData> = _gyroscopeData

    private val _gpsData = mutableStateOf(GpsData(0.0, 0.0, 0f, 0L))
    val gpsData: State<GpsData> = _gpsData

    private val _totalAcceleration = mutableStateOf(0f)
    val totalAcceleration: State<Float> = _totalAcceleration

    private val _totalAngularVelocity = mutableStateOf(0f)
    val totalAngularVelocity: State<Float> = _totalAngularVelocity

    private val _isGoodRoad = mutableStateOf(true)
    val isGoodRoad: State<Boolean> = _isGoodRoad

    private val _isRecording = mutableStateOf(false)
    val isRecording: State<Boolean> = _isRecording

    val recordedData = mutableListOf<String>()

    fun setAccelerometerData(x: Float, y: Float, z: Float, timestamp: Long) {
        _accelerometerData.value = SensorData(x, y, z, timestamp)
        _totalAcceleration.value = sqrt(x * x + y * y + z * z)
    }

    fun setGyroscopeData(x: Float, y: Float, z: Float, timestamp: Long) {
        _gyroscopeData.value = SensorData(x, y, z, timestamp)
        _totalAngularVelocity.value = sqrt(x * x + y * y + z * z)
    }

    fun setGpsData(latitude: Double, longitude: Double, speed: Float, timestamp: Long) {
        _gpsData.value = GpsData(latitude, longitude, speed, timestamp)
    }

    fun toggleRoadCondition() {
        _isGoodRoad.value = !_isGoodRoad.value
    }

    fun toggleRecording() {
        _isRecording.value = !_isRecording.value
        if (_isRecording.value) {
            recordedData.clear()
        }
    }
}
