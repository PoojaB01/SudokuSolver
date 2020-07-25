import 'package:http/http.dart' as http;

Future findsolution(url) async {
  http.Response response = await http.get(url);
  print("made");
  return response.body;
}
