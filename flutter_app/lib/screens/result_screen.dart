import 'dart:io';

import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../widgets/result_card.dart';

class ResultScreen extends StatefulWidget {
  final File imageFile;
  final Map<String, dynamic> personalization;

  ResultScreen({required this.imageFile, required this.personalization});

  @override
  _ResultScreenState createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  bool _loading = true;
  Map<String, dynamic>? _result;

  @override
  void initState() {
    super.initState();
    _sendToServer();
  }

  Future<void> _sendToServer() async {
    try {
      final res = await ApiService.uploadImageAndData(widget.imageFile, widget.personalization);
      setState(() {
        _result = res;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _result = {'error': e.toString()};
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Result')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: _loading
            ? Center(child: CircularProgressIndicator())
            : _result == null
                ? Center(child: Text('No result'))
                : _result!.containsKey('error')
                    ? Center(child: Text('Error: ${_result!['error']}'))
                    : ResultCard(result: _result!),
      ),
    );
  }
}
