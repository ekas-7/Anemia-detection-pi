import 'package:flutter/material.dart';
import 'camera_screen.dart';

class PersonalizationForm extends StatefulWidget {
  @override
  _PersonalizationFormState createState() => _PersonalizationFormState();
}

class _PersonalizationFormState extends State<PersonalizationForm> {
  final _formKey = GlobalKey<FormState>();
  final Map<String, dynamic> _data = {
    'familyHistory': '',
    'kime': '',
    'diet': 'Vegetarian',
    'sickleCell': false,
  };

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Personalization')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: ListView(
            children: [
              TextFormField(
                decoration: InputDecoration(labelText: 'Family history (notes)'),
                onSaved: (v) => _data['familyHistory'] = v ?? '',
              ),
              SizedBox(height: 12),
              TextFormField(
                decoration: InputDecoration(labelText: 'KIME (medical notes / hemoglobin)'),
                onSaved: (v) => _data['kime'] = v ?? '',
              ),
              SizedBox(height: 12),
              DropdownButtonFormField<String>(
                initialValue: _data['diet'] as String,
                decoration: InputDecoration(labelText: 'Dietary Pattern'),
                items: ['Vegetarian', 'Non-Vegetarian']
                    .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                    .toList(),
                onChanged: (v) => setState(() => _data['diet'] = v ?? 'Vegetarian'),
              ),
              SizedBox(height: 12),
              SwitchListTile(
                title: Text('Sickle cell predisposition'),
                value: _data['sickleCell'],
                onChanged: (v) => setState(() => _data['sickleCell'] = v),
              ),
              SizedBox(height: 20),
              ElevatedButton(
                child: Text('Continue to Camera'),
                onPressed: () {
                  _formKey.currentState?.save();
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (_) => CameraScreen(personalization: _data),
                    ),
                  );
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}
