package fit.magic.cv.repcounter

import android.util.Log
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import fit.magic.cv.PoseLandmarkerHelper
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.min

// Key joints for angle calculations
class JointSet
{
    var leftHip: NormalizedLandmark
    var leftKnee: NormalizedLandmark
    var leftAnkle: NormalizedLandmark
    var rightHip: NormalizedLandmark
    var rightKnee: NormalizedLandmark
    var rightAnkle: NormalizedLandmark

    constructor(lh: NormalizedLandmark, lk: NormalizedLandmark, la: NormalizedLandmark,
                rh: NormalizedLandmark, rk: NormalizedLandmark, ra: NormalizedLandmark)
    {
        leftHip = lh
        leftKnee = lk
        leftAnkle = la
        rightHip = rh
        rightKnee = rk
        rightAnkle = ra
    }
}

class ExerciseRepCounterImpl : ExerciseRepCounter()
{
    private val depthSnapshots = mutableListOf<Float>()
    private val stanceArray = mutableListOf<Int>()
    private val filteredAngles = mutableListOf<Float>()

    private var previousFilteredValue = 0f
    private var positionReached = false
    private var repetitionFlag = false

    // Median smoothing over the last N frames
    private val smoothingWindow = 5
    private val landmarkHistory = mutableMapOf<Int, MutableList<NormalizedLandmark>>()

    // Threshold values for angles ( with a slight margin of error )
    private val triggerThreshold = 118f
    private val bottomOutThreshold = 88f

    private var lastMessage: String? = null

    override fun setResults(resultData: PoseLandmarkerHelper.ResultBundle)
    {
        val firstPose = resultData.results.firstOrNull() ?: run {
            sendFeedbackMessage("No joint data retrieved.")
            return
        }

        val foundLandmarks = firstPose.landmarks()

        try
        {
            val stableLandmarks = applyMedianSmoothing(foundLandmarks[0])
            val groupedJoints = bundleRequiredJoints(stableLandmarks)
            val kneeAngleData = deriveKneeAngles(groupedJoints)
            analyzeAngles(kneeAngleData)
        }
        catch (e: Exception)
        {
            Log.e("ExerciseRepCounterImpl", "Error processing results", e)
        }
    }

    // ------------------- MEDIAN SMOOTHING -------------------
    private fun applyMedianSmoothing(rawPoints: List<NormalizedLandmark>): List<NormalizedLandmark>
    {
        rawPoints.forEachIndexed { index, point ->
            if (!landmarkHistory.containsKey(index)) {
                landmarkHistory[index] = mutableListOf()
            }
            val buf = landmarkHistory[index]!!
            buf.add(point)
            if (buf.size > smoothingWindow) {
                buf.removeAt(0)
            }
        }

        return rawPoints.mapIndexed { idx, _ ->
            val hist = landmarkHistory[idx]!!
            val medX = medianOf(hist.map { it.x() })
            val medY = medianOf(hist.map { it.y() })
            val medZ = medianOf(hist.map { it.z() })

            NormalizedLandmark.create(
                medX, medY, medZ,
                hist.last().presence(),
                hist.last().visibility()
            )
        }
    }

    private fun medianOf(vals: List<Float>): Float
    {
        val sortedVals = vals.sorted()
        return if (sortedVals.size % 2 == 1) {
            sortedVals[sortedVals.size / 2]
        } else {
            (sortedVals[sortedVals.size/2 - 1] + sortedVals[sortedVals.size/2]) / 2f
        }
    }

    // ------------------- JOINTS -------------------
    private fun bundleRequiredJoints(points: List<NormalizedLandmark>): JointSet
    {
        val leftHip = points[23]
        val leftKnee = points[25]
        val leftAnkle = points[27]
        val rightHip = points[24]
        val rightKnee = points[26]
        val rightAnkle = points[28]

        return JointSet(leftHip, leftKnee, leftAnkle, rightHip, rightKnee, rightAnkle)
    }

    // ------------------- ANGLE CALCULATIONS -------------------
    private fun deriveKneeAngles(joints: JointSet): Map<String, Float>
    {
        val angleMap = mutableMapOf<String, Float>()
        angleMap["LEFT"] = computeAngle(joints.leftHip, joints.leftKnee, joints.leftAnkle)
        angleMap["RIGHT"] = computeAngle(joints.rightHip, joints.rightKnee, joints.rightAnkle)
        return angleMap
    }

    private fun computeAngle(a: NormalizedLandmark, b: NormalizedLandmark, c: NormalizedLandmark): Float
    {
        val rad = atan2(c.y() - b.y(), c.x() - b.x()) - atan2(a.y() - b.y(), a.x() - b.x())
        var deg = abs((rad * 180f) / PI.toFloat())
        if (deg > 180f) deg = 360f - deg
        return deg
    }

    // ------------------- LOGIC & UI -------------------
    private fun analyzeAngles(angles: Map<String, Float>)
    {
        val leftVal = angles["LEFT"] ?: return
        val rightVal = angles["RIGHT"] ?: return

        val currentDepth = min(leftVal, rightVal)
        depthSnapshots.add(currentDepth)

        val progressLevel = max(
            0f,
            min((triggerThreshold - currentDepth) / (triggerThreshold - bottomOutThreshold), 1f)
        )

        stanceArray.add(if (currentDepth < triggerThreshold) 1 else 0)

        if (stanceArray.last() == 1 && currentDepth < bottomOutThreshold) {
            if (depthSnapshots.size >= 3
                && depthSnapshots[depthSnapshots.size - 2] < depthSnapshots.last()
                && depthSnapshots[depthSnapshots.size - 3] > depthSnapshots[depthSnapshots.size - 2]
            ) {
                positionReached = true
            }
        } else if (stanceArray.last() == 0) {
            positionReached = false
        }

        // Smooth angles
        if (stanceArray.last() == 1 && !positionReached) {
            previousFilteredValue = max(progressLevel, previousFilteredValue)
            filteredAngles.add(previousFilteredValue)
        } else if (stanceArray.last() == 1 && positionReached) {
            previousFilteredValue = min(progressLevel, previousFilteredValue)
            filteredAngles.add(previousFilteredValue)
            if (!repetitionFlag) {
                incrementRepCount()
                repetitionFlag = true
            }
        } else if (stanceArray.last() == 0) {
            previousFilteredValue = 0f
            filteredAngles.add(previousFilteredValue)
            repetitionFlag = false
        }

        // Simple feedback messages based on progress
        val progressPercent = (filteredAngles.last() * 100).toInt()

        val newMessage = when {
            repetitionFlag && progressPercent == 100 -> {

                "Start lunge"
            }
            progressPercent <= 20 -> {
                "Start lunge"
            }
            progressPercent < 80 -> {
                "Lunge further"
            }
            progressPercent < 100 -> {
                "Nearly there"
            }
            else -> {

                null
            }
        }

        if (newMessage != null && newMessage != lastMessage) {
            sendFeedbackMessage(newMessage)
            lastMessage = newMessage
        }

        sendProgressUpdate(filteredAngles.last())
        debugLog(filteredAngles.last(), stanceArray.last(), positionReached, currentDepth)
    }

    private fun debugLog(value: Float, stance: Int, reached: Boolean, depthVal: Float)
    {
        Log.d("RepCounterDebug", "Progress:$value, Active:$stance, Deepest:$reached, Depth:$depthVal")
    }
}