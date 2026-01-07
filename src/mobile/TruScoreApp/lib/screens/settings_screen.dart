import 'package:flutter/material.dart';
import 'package:settings_ui/settings_ui.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/api_client.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final TextEditingController _ipController = TextEditingController();
  String _currentIp = '';

  @override
  void initState() {
    super.initState();
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _currentIp = prefs.getString('custom_api_ip') ?? 'http://10.0.2.2:8009';
    _ipController.text = _currentIp;
    });
  }

  Future<void> _saveIp(String ip) async {
    if (ip.isEmpty) return;
    String formatted = ip;
    if (!formatted.startsWith('http')) {
      formatted = 'http://$formatted';
    }

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('custom_api_ip', formatted);
    setState(() {
      _currentIp = formatted;
    });
    ApiClient.updateBaseUrl(formatted);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      backgroundColor: Colors.black, // Match app theme
      body: SettingsList(
        sections: [
          SettingsSection(
            title: const Text('Server Configuration'),
            tiles: <SettingsTile>[
              SettingsTile.navigation(
                leading: const Icon(Icons.computer),
                title: const Text('Server IP Address'),
                value: Text(_currentIp),
                onPressed: (context) {
                  _showIpDialog(context);
                },
              ),
            ],
          ),
          SettingsSection(
            title: const Text('About'),
            tiles: <SettingsTile>[
              SettingsTile(
                leading: const Icon(Icons.info),
                title: const Text('App Version'),
                value: const Text('1.0.0'),
              ),
            ],
          ),
        ],
      ),
    );
  }

  void _showIpDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Set Server IP'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              "Enter the IP shown on your desktop app (e.g. 192.168.1.x:8009)",
              style: TextStyle(fontSize: 12, color: Colors.grey),
            ),
            const SizedBox(height: 10),
            TextField(
              controller: _ipController,
              decoration: const InputDecoration(
                hintText: 'http://192.168.1.5:8009',
                labelText: 'API URL',
                border: OutlineInputBorder(),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              _saveIp(_ipController.text.trim());
              Navigator.pop(context);
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }
}
