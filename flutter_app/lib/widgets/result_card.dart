import 'package:flutter/material.dart';

class ResultCard extends StatelessWidget {
  final Map<String, dynamic> result;
  ResultCard({required this.result});

  @override
  Widget build(BuildContext context) {
    final label = result['label'] ?? 'Unknown';
    final confidence = (result['confidence'] != null) ? (result['confidence'] as num).toDouble() : 0.0;
    final severity = result['severity'] ?? '';

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Prediction', style: Theme.of(context).textTheme.subtitle1),
            SizedBox(height: 8),
            Text(label, style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
            SizedBox(height: 8),
            LinearProgressIndicator(value: confidence.clamp(0.0, 1.0)),
            SizedBox(height: 8),
            Text('Confidence: ${(confidence * 100).toStringAsFixed(1)}%'),
            if (severity != null && severity != '') ...[
              SizedBox(height: 8),
              Text('Severity: $severity'),
            ]
          ],
        ),
      ),
    );
  }
}
