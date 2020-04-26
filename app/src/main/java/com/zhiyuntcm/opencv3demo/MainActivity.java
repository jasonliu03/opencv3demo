package com.zhiyuntcm.opencv3demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }


    ImageView imgView;
    ImageView imgViewAfter;
    private Button mButton_animation;
    private Bitmap img;
    private int[] pix;

    private int w,h;

    public class GetJsonDataUtil {

        public String getJson(String fileName) {

            StringBuilder stringBuilder = new StringBuilder();
            try {
                AssetManager assetManager = getAssets();
                BufferedReader bf = new BufferedReader(new InputStreamReader(
                        assetManager.open(fileName)));
                String line;
                while ((line = bf.readLine()) != null) {
                    stringBuilder.append(line);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            return stringBuilder.toString();
        }
    }

    public String GetImageStr(String fileName) {
        InputStream in = null;
        byte[] data = null;
        // 读取图片字节数组
        try {
            AssetManager assetManager = getAssets();
            in = assetManager.open(fileName);
            data = new byte[in.available()];
            in.read(data);
            in.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        // 对字节数组Base64编码
        String strBase64 = Base64.encodeToString(data, Base64.DEFAULT);
        // save base64 str to file
        FileWriter fwriter = null;
        try {
            File root = Environment.getExternalStorageDirectory();
            File myFile=new File(root+"/json.txt");
            Log.i("gebilaolitou", "myFile=" + (root+"/json.txt"));
            fwriter = new FileWriter(myFile);
            fwriter.write(strBase64);
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                if(fwriter != null){
                    fwriter.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        // 返回Base64编码过的字节数组字符串
        return strBase64;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);



        String base64str = GetImageStr("cc06_600.jpg");
        byte[] b = Base64.decode(base64str.getBytes(), Base64.DEFAULT);
        img = BitmapFactory.decodeByteArray(b, 0, b.length);

        w = img.getWidth();
        h = img.getHeight();
        Log.i("gebilaolitou", "w,h="+w+h);
        pix = new int[w * h];
        img.getPixels(pix, 0, w, 0, 0, w, h);

        imgView = (ImageView) this.findViewById(R.id.imageView);
        imgView.setImageBitmap(img);

        imgViewAfter = (ImageView) this.findViewById(R.id.imageViewAfter);


        // Example of a call to a native method
//        TextView tv = findViewById(R.id.sample_text);
//        tv.setText(stringFromJNI());


        mButton_animation = (Button) findViewById(R.id.animation);
        mButton_animation.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v) {
                //animation
                long current = System.currentTimeMillis();

                int[] resultInt = animationFaceMask(pix, w, h);

                Bitmap resultImg = Bitmap.createBitmap(w, h, Bitmap.Config.RGB_565);
                resultImg.setPixels(resultInt, 0, w, 0, 0, w, h);
                long performance = System.currentTimeMillis() - current;
                imgViewAfter.setImageBitmap(resultImg);
            }
        });
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();

    public static native int[] animationFaceMask(int[] pixels,int w,int h);
}
