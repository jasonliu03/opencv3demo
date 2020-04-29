package com.zhiyuntcm.opencv3demo;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Environment;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
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
import java.io.FileOutputStream;
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

    private InputStream is;
    private FileOutputStream os = null;
    private File cascadeDir;
    private File mCascadeFile;

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

    public void loadModelFile(InputStream is, String filename)
    {
        cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
        mCascadeFile = new File(cascadeDir, filename);
        try {
            os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
//            cascadeDir.delete();
        } catch (IOException e) {
            e.printStackTrace();
            Log.e("gebilaolitou", "Failed to load cascade. Exception thrown: " + e);
        }
    }

    public void faceDiagnosis()
    {
        is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt2);
        loadModelFile(is, "haarcascade_frontalface_alt2.xml");
        String cascadeFace = mCascadeFile.getAbsolutePath();
        is = getResources().openRawResource(R.raw.svm_facecomplexion);
        loadModelFile(is, "svm_facecomplexion.xml");
        String svmFace = mCascadeFile.getAbsolutePath();
        is = getResources().openRawResource(R.raw.facegloss_model);
        loadModelFile(is, "facegloss_model.xml");
        String glossFace = mCascadeFile.getAbsolutePath();
        is = getResources().openRawResource(R.raw.haarcascade_lip);
        loadModelFile(is, "haarcascade_lip.xml");
        String cascadeLip = mCascadeFile.getAbsolutePath();
        is = getResources().openRawResource(R.raw.svm_lipcolor);
        loadModelFile(is, "svm_lipcolor.xml");
        String svmLip = mCascadeFile.getAbsolutePath();

        TextView tv = findViewById(R.id.sample_text);
        long current2 = System.currentTimeMillis();
        String rstFace = tcmFacePro(pix, w, h, cascadeFace, svmFace, glossFace, cascadeLip, svmLip);
        Log.i("gebilaolitou", "rstFace="+rstFace);
        tv.setText(rstFace);
        long performance2 = System.currentTimeMillis() - current2;
        Log.i("gebilaolitou","time_faceDiagnosis="+performance2);
    }

    public void tongueDiagnosis()
    {
        is = getResources().openRawResource(R.raw.haarcascade_tongue);
        loadModelFile(is, "haarcascade_tongue.xml");
        String cascadeFaceS = mCascadeFile.getAbsolutePath();
        is = getResources().openRawResource(R.raw.svm_tonguecoatcolor);
        loadModelFile(is, "svm_tonguecoatcolor.xml");
        String ccolorClassifierPathName = mCascadeFile.getAbsolutePath();
        is = getResources().openRawResource(R.raw.svm_tonguecoatthickness);
        loadModelFile(is, "svm_tonguecoatthickness.xml");
        String bohouClassifierPathName = mCascadeFile.getAbsolutePath();
        is = getResources().openRawResource(R.raw.svm_tonguefatthin);
        loadModelFile(is, "svm_tonguefatthin.xml");
        String panClassifierPathName = mCascadeFile.getAbsolutePath();

        TextView tv = findViewById(R.id.sample_text);
        long current2 = System.currentTimeMillis();
        String rstTongue = tcmTonguePro(pix, w, h, cascadeFaceS, ccolorClassifierPathName, bohouClassifierPathName, panClassifierPathName);
        Log.i("gebilaolitou", "rstTongue="+rstTongue);
        tv.setText(rstTongue);
        long performance2 = System.currentTimeMillis() - current2;
        Log.i("gebilaolitou","time_tongueDiagnosis="+performance2);
    }

    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE};
    //请求状态码
    private static int REQUEST_PERMISSION_CODE = 1;
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_PERMISSION_CODE) {
            for (int i = 0; i < permissions.length; i++) {
                Log.i("gebilaolitou", "申请的权限为：" + permissions[i] + ",申请结果：" + grantResults[i]);
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.LOLLIPOP) {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, PERMISSIONS_STORAGE, REQUEST_PERMISSION_CODE);
            }
        }


        String base64str = GetImageStr("neg1.jpg");
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

//        faceDiagnosis();
        tongueDiagnosis();

        mButton_animation = (Button) findViewById(R.id.animation);
        mButton_animation.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v) {
                //animation
                long current = System.currentTimeMillis();

                int[] resultInt = animationTongueMask(pix, w, h);
//                int[] resultInt = {0};

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

//    public static native int[] animationFaceMask(int[] pixels,int w,int h);
    public static native int[] animationTongueSegmentation(int[] pixels,int w,int h);
    public static native int[] animationTongueMask(int[] pixels,int w,int h);
    public static native String tcmFacePro(int[] pixels,int w,int h,String cascadeFileName, String svmFace, String glossFace, String cascadeLip, String svmLip);
    public static native String tcmTonguePro(int[] pixels,int w,int h,String cascadeFileName, String ccolorClassifierPathName, String bohouClassifierPathName, String panClassifierPathName);

}
