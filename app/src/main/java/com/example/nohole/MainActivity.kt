package com.example.nohole

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.graphics.drawable.BitmapDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.android.volley.Request
import com.android.volley.toolbox.JsonObjectRequest
import com.android.volley.toolbox.Volley
import org.json.JSONException
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker


class MainActivity : AppCompatActivity() {

    private lateinit var mapView: MapView
    private val handler = Handler(Looper.getMainLooper())
    private val pollIntervalMs: Long = 5000   // every 5 sec

    // ⚠️ Change this to your backend URL
    private val SERVER_URL = "http://10.53.159.154:5000/get_latest"


    private val LOCATION_PERMISSION_CODE = 1001

    private val pollRunnable = object : Runnable {
        override fun run() {
            fetchPredictionFromServer()
            handler.postDelayed(this, pollIntervalMs)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // osmdroid config
        Configuration.getInstance().load(
            applicationContext,
            applicationContext.getSharedPreferences("osmdroid", MODE_PRIVATE)
        )

        setContentView(R.layout.activity_main)

        mapView = findViewById(R.id.mapView)
        mapView.setTileSource(TileSourceFactory.MAPNIK)

        mapView.controller.setZoom(16.0)
        mapView.controller.setCenter(GeoPoint(28.6139, 77.2090))
// 🔥 TEST MARKER WITHOUT BACKEND
        //Handler(Looper.getMainLooper()).postDelayed({
          //  addColoredMarker(28.6139, 77.2090, "Good")   // Green marker
            //addColoredMarker(28.6200, 77.2100, "Bad")    // Red marker
        //}, 2000)


        checkAndRequestPermissions()

       handler.post(pollRunnable)
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacks(pollRunnable)
    }

    private fun checkAndRequestPermissions() {
        val needed = mutableListOf<String>()
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            needed.add(Manifest.permission.ACCESS_FINE_LOCATION)
        }
        if (needed.isNotEmpty()) {
            ActivityCompat.requestPermissions(
                this,
                needed.toTypedArray(),
                LOCATION_PERMISSION_CODE
            )
        }
    }

    private fun fetchPredictionFromServer() {
        val queue = Volley.newRequestQueue(this)

        val request = JsonObjectRequest(
            Request.Method.GET,
            SERVER_URL,
            null,
            { response ->
                try {
                    val lat = response.getDouble("latitude")
                    val lon = response.getDouble("longitude")
                    val condition = response.getString("road_condition")
                    addColoredMarker(lat, lon, condition)
                } catch (e: JSONException) {
                    Log.e("PRED_PARSE", "JSON parse error: ${e.message}")
                }
            },
            { error ->
                Log.e("PRED_FETCH", "Error fetching: ${error.message}")
            }
        )

        queue.add(request)
    }

    private fun addColoredMarker(lat: Double, lon: Double, condition: String) {
        val marker = Marker(mapView)
        marker.position = GeoPoint(lat, lon)
        marker.title = condition

        val color = if (condition.equals("Good", ignoreCase = true))
            Color.parseColor("#2E7D32")  // green
        else
            Color.parseColor("#C62828")  // red

        marker.icon = getColoredCircleDrawable(color)
        marker.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)

        mapView.overlays.add(marker)
        mapView.invalidate()
    }

    private fun getColoredCircleDrawable(color: Int): BitmapDrawable {
        val size = (30 * resources.displayMetrics.density).toInt()
        val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)

        paint.color = color
        canvas.drawCircle(size / 2f, size / 2f, size / 2f, paint)

        paint.color = Color.WHITE
        canvas.drawCircle(size / 2f, size / 2f, size / 4f, paint)

        return BitmapDrawable(resources, bitmap)
    }
}
