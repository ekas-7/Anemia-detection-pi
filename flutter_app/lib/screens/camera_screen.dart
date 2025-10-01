import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'result_screen.dart';

class CameraScreen extends StatefulWidget {
  final Map<String, dynamic>? personalization;
  CameraScreen({this.personalization});

  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  File? _image;
  final ImagePicker _picker = ImagePicker();

  Future<void> _takePhoto() async {
    final XFile? photo = await _picker.pickImage(source: ImageSource.camera, imageQuality: 80);
    if (photo != null) setState(() => _image = File(photo.path));
  }

  Future<void> _chooseFromGallery() async {
    final XFile? photo = await _picker.pickImage(source: ImageSource.gallery, imageQuality: 80);
    if (photo != null) setState(() => _image = File(photo.path));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Capture Conjunctiva')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Expanded(
              child: Center(
                child: _image == null
                    ? Text('No image selected')
                    : Image.file(_image!),
              ),
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  icon: Icon(Icons.camera_alt),
                  label: Text('Take Photo'),
                  onPressed: _takePhoto,
                ),
                ElevatedButton.icon(
                  icon: Icon(Icons.photo_library),
                  label: Text('Gallery'),
                  onPressed: _chooseFromGallery,
                ),
              ],
            ),
            SizedBox(height: 12),
            ElevatedButton(
              child: Text('Send to Server'),
              onPressed: _image == null
                  ? null
                  : () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => ResultScreen(
                            imageFile: _image!,
                            personalization: widget.personalization ?? {},
                          ),
                        ),
                      );
                    },
            ),
          ],
        ),
      ),
    );
  }
}
