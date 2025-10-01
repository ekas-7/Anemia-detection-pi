import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

class ApiService {
  // Update this to your server endpoint
  static const String _endpoint = 'http://10.0.2.2:5000/predict';

  static Future<Map<String, dynamic>> uploadImageAndData(File image, Map<String, dynamic> data) async {
    final uri = Uri.parse(_endpoint);
    final request = http.MultipartRequest('POST', uri);
    request.files.add(await http.MultipartFile.fromPath('image', image.path));
    request.fields['data'] = jsonEncode(data);

    final streamed = await request.send();
    final resp = await http.Response.fromStream(streamed);
    if (resp.statusCode == 200) {
      return jsonDecode(resp.body) as Map<String, dynamic>;
    } else {
      throw Exception('Server error: ${resp.statusCode} ${resp.reasonPhrase}');
    }
  }
}
