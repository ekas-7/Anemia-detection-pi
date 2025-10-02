import 'package:flutter/material.dart';
import 'personalization_form.dart';

class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Anemia Detection'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'Personalized anemia screening',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            SizedBox(height: 12),
            Text(
              'Provide a few details and capture an image of the lower eyelid conjunctiva.',
            ),
            SizedBox(height: 24),
            ElevatedButton.icon(
              icon: Icon(Icons.person),
              label: Text('Enter Personalization Data'),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => PersonalizationForm()),
                );
              },
            ),
            SizedBox(height: 12),
            ElevatedButton.icon(
              icon: Icon(Icons.camera_alt),
              label: Text('Open Camera & Capture'),
              onPressed: () {
                Navigator.pushNamed(context, '/camera');
              },
            ),
          ],
        ),
      ),
    );
  }
}
