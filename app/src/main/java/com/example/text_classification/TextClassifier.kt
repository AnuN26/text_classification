package com.example.text_classification

import android.content.Context
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.io.IOException

class TextClassifier(context: Context, jsonFilename: String, inputMaxLen : Int) {
    private var context : Context? = context

    // Filename for the exported vocab ( .json )
    private var filename : String? = jsonFilename

    // Max length of the input sequence for the given model.
    private var maxlen : Int = inputMaxLen

    private var vocabData : HashMap< String , Int >? = null

    // Load the contents of the vocab ( see assets/word_dict.json )
    private fun loadJSONFromAsset(filename : String? ): String? {
        var json: String?
        try {
            val inputStream = filename?.let { context!!.assets.open(it) }
            val size = inputStream?.available()
            val buffer = size?.let { ByteArray(it) }
            if (inputStream != null) {
                inputStream.read(buffer)
            }
            if (inputStream != null) {
                inputStream.close()
            }
            json = buffer?.let { String(it) }
        }
        catch (ex: IOException) {
            ex.printStackTrace()
            return null
        }
        return json
    }

    fun processVocab( callback: VocabCallback ) {
        CoroutineScope( Dispatchers.Main ).launch {
            loadVocab( callback , loadJSONFromAsset( filename )!! )
        }
    }

    // Tokenize the given sentence
    fun tokenize ( message : String ): IntArray {
        val parts : List<String> = message.split(" " ) //splitting the message
        val tokenizedMessage = ArrayList<Int>()
        for ( part in parts ) {
            if (part.trim() != ""){
                var index : Int? = 0
                index = if ( vocabData!![part] == null ) {
                    0
                } else{
                    vocabData!![part]
                }
                tokenizedMessage.add( index!! )
            }
        }
        return tokenizedMessage.toIntArray()
    }

    // Pad the given sequence to maxlen with zeros.
    fun padSequence ( sequence : IntArray ) : IntArray {
        val maxlen = this.maxlen
        if ( sequence.size > maxlen ) {
            return sequence.sliceArray( 0..maxlen )
        }
        else if ( sequence.size < maxlen ) {
            val array = ArrayList<Int>()
            array.addAll( sequence.asList() )
            for ( i in array.size until maxlen ){
                array.add(0)
            }
            return array.toIntArray()
        }
        else{
            return sequence
        }
    }


    interface VocabCallback {
        fun onVocabProcessed()
    }

    private fun loadVocab(callback : VocabCallback, json : String )  {
        with( Dispatchers.Default ) {// it will helps to parsing the json.
            val jsonObject = JSONObject( json )
            val iterator : Iterator<String> = jsonObject.keys()
            val data = HashMap< String , Int >()
            while ( iterator.hasNext() ) {
                val key = iterator.next()
                data[key] = jsonObject.get( key ) as Int
            }
            with( Dispatchers.Main ){//run with android main thread.
                // it will helps to interacting with the UI
                vocabData = data
                callback.onVocabProcessed()
            }
        }
    }

}