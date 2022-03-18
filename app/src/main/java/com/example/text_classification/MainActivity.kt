package com.example.text_classification

import android.app.ProgressDialog
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.text.TextUtils
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    // Name of TFLite model ( in /assets folder ).
    private val MODEL_ASSETS_PATH = "model.tflite"

    // Max Length of input sequence. The input shape for the model will be ( None , INPUT_MAXLEN ).
    private val INPUT_MAXLEN = 171

    private var tfLiteInterpreter : Interpreter? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val message_text: TextView =findViewById(R.id.message_text)
        val result_text: TextView =findViewById(R.id.result_text)
        val classifyButton: Button =findViewById(R.id.classifyButton)

        // Init the classifier.
        val classifier = TextClassifier( this , "word_dict.json" , INPUT_MAXLEN )
        // Init TFLiteInterpreter
        tfLiteInterpreter = Interpreter( loadModelFile() )

        // Start vocab processing, show a ProgressDialog to the user.
        val progressDialog = ProgressDialog( this )
        progressDialog.setMessage( "Parsing word_dict.json ..." )
        progressDialog.setCancelable( false )
        progressDialog.show()
        classifier.processVocab( object: TextClassifier.VocabCallback {
            override fun onVocabProcessed() {
                // Processing done, dismiss the progressDialog.
                progressDialog.dismiss()
            }
        })

        classifyButton.setOnClickListener {

            val message = message_text.text.toString().toLowerCase().trim()
            if ( !TextUtils.isEmpty( message ) ){
                // Tokenize and pad the given input text.
                val tokenizedMessage = classifier.tokenize( message )
                val paddedMessage = classifier.padSequence( tokenizedMessage )

                val results = classifySequence( paddedMessage )
                val class1 = results[0]
                val class2 = results[1]
                result_text.text = "SPAM : $class2\nNOT SPAM : $class1 "
            }
            else{
                Toast.makeText( this, "Please enter a message.", Toast.LENGTH_LONG).show();
            }

        }

    }
    @Throws(IOException::class)
    private fun loadModelFile(): MappedByteBuffer {
        val assetFileDescriptor = assets.openFd(MODEL_ASSETS_PATH)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel //get unique FileChannel
        val startOffset = assetFileDescriptor.startOffset // length of the file
        val declaredLength = assetFileDescriptor.declaredLength // when length has not been declared - constant value is 1
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength) //load the tensor flow model from internal storage
    }

    // Perform inference, given the input sequence.
    private fun classifySequence (sequence : IntArray ): FloatArray {
        // Input shape -> ( 1 , INPUT_MAXLEN )
        val inputs : Array<FloatArray> = arrayOf( sequence.map { it.toFloat() }.toFloatArray() )
        // Output shape -> ( 1 , 2 ) ( as numClasses = 2 )
        val outputs : Array<FloatArray> = arrayOf( FloatArray( 2 ) )
        tfLiteInterpreter?.run( inputs , outputs )
        return outputs[0]
    }

}