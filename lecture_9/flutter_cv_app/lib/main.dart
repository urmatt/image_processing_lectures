import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:typed_data';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLOv8 Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'YOLOv8 CV Integration'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? _image;
  Uint8List? _processedImage;
  bool _isLoading = false;
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage() async {
    final XFile? pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _processedImage = null; // Clear previous result
      });
    }
  }

  Future<void> _processImage() async {
    if (_image == null) return;

    setState(() {
      _isLoading = true;
    });

    try {
      // Подставьте IP адрес вашего компьютера в локальной сети, 
      // если запускаете на реальном устройстве, или 10.0.2.2 для эмулятора Android.
      // 127.0.0.1 подойдет для macOS/Windows/Linux десктоп приложения.
      final url = Uri.parse('http://127.0.0.1:8000/predict');
      
      final request = http.MultipartRequest('POST', url)
        ..files.add(await http.MultipartFile.fromPath('file', _image!.path));

      final response = await request.send();

      if (response.statusCode == 200) {
        final bytes = await response.stream.toBytes();
        setState(() {
          _processedImage = bytes;
        });
      } else {
        _showError('Server error: ${response.statusCode}');
      }
    } catch (e) {
      _showError('Error connecting to server: $e');
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              if (_image != null) ...[
                const Text('Оригинал:', style: TextStyle(fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                Image.file(_image!, height: 300, fit: BoxFit.contain),
                const SizedBox(height: 16),
                ElevatedButton.icon(
                  onPressed: _isLoading ? null : _processImage,
                  icon: _isLoading 
                      ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2)) 
                      : const Icon(Icons.cloud_upload),
                  label: const Text('Send на сервер (YOLOv8)'),
                ),
                const SizedBox(height: 16),
              ],
              if (_processedImage != null) ...[
                const Text('Результат:', style: TextStyle(fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                Image.memory(_processedImage!, height: 300, fit: BoxFit.contain),
              ],
              if (_image == null)
                const SizedBox(
                  height: 300,
                  child: Center(
                    child: Text('Выберите изображение для распознавания'),
                  ),
                ),
            ],
          ),
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _pickImage,
        tooltip: 'Выбрать фото',
        child: const Icon(Icons.add_photo_alternate),
      ),
    );
  }
}
