import 'package:flutter/material.dart';
import 'screens/home_screen.dart';

void main() {
  runApp(AnemiaApp());
}

class AnemiaApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final primaryGreen = Color(0xFF2E7D32);
    final lightGreen = Color(0xFF66BB6A);

    return MaterialApp(
      title: 'Anemia Detection',
      theme: ThemeData(
        primaryColor: primaryGreen,
        colorScheme: ColorScheme.fromSwatch(primarySwatch: Colors.green)
            .copyWith(secondary: lightGreen),
        scaffoldBackgroundColor: Colors.white,
        appBarTheme: AppBarTheme(
          backgroundColor: primaryGreen,
          foregroundColor: Colors.white,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: primaryGreen,
            foregroundColor: Colors.white,
          ),
        ),
      ),
      home: HomeScreen(),
    );
  }
}
