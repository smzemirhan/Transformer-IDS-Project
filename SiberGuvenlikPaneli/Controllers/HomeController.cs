using Microsoft.AspNetCore.Mvc;
using Newtonsoft.Json;
using System.Text;
using SiberGuvenlikPaneli.Models; // Model dosyanın olduğu namespace
using System.Diagnostics;

namespace SiberGuvenlikPaneli.Controllers
{
    public class HomeController : Controller
    {
        // Geçmişi hafızada tutmak için liste
        private static List<LogEntry> _history = new List<LogEntry>();

        public IActionResult Index()
        {
            ViewBag.History = _history;
            return View(new TrafficData());
        }

        [HttpPost]
        public async Task<IActionResult> Analyze(TrafficData data)
        {
            // 1. GÜVENLİ VERİ PAKETLEME (Null değerleri engelleme)
            var payload = new
            {
                log_data = new
                {
                    proto = data.Proto ?? "tcp",
                    service = data.Service ?? "http",
                    duration = data.Duration, // Double olduğu için null olamaz
                    conn_state = data.ConnState ?? "SF",
                    history = data.History ?? "ShADadFf",
                    orig_pkts = data.OrigPkts,
                    orig_ip_bytes = data.OrigIpBytes,
                    resp_pkts = data.RespPkts,
                    resp_ip_bytes = data.RespIpBytes
                }
            };

            using (var client = new HttpClient())
            {
                // 5 saniye içinde cevap gelmezse hata ver
                client.Timeout = TimeSpan.FromSeconds(5);

                try
                {
                    var jsonContent = new StringContent(JsonConvert.SerializeObject(payload), Encoding.UTF8, "application/json");

                    // --- KALI LINUX IP ADRESİ ---
                    
                    string kaliIp = "192.168.56.103";

                    var response = await client.PostAsync($"http://{kaliIp}:8000/predict", jsonContent);

                    if (response.IsSuccessStatusCode)
                    {
                        var responseString = await response.Content.ReadAsStringAsync();

                        // Dinamik olarak çöz 
                        dynamic result = JsonConvert.DeserializeObject(responseString);

                        // --- API HATASI KONTROLÜ ---
                        if (result.error != null)
                        {
                            throw new Exception("Python Hatası: " + (string)result.error);
                        }

                        // Verileri C# modeline aktar (Null gelirse varsayılan değer ata)
                        data.Prediction = (string)result.prediction ?? "Bilinmiyor";
                        data.Status = (string)result.status ?? "TEHLİKELİ";
                        data.Confidence = (double?)result.confidence ?? 0.0;
                        data.IsAnalyzed = true;

                        // --- RENK VE AKSİYON MANTIĞI ---
                        if (data.Status == "GÜVENLİ")
                        {
                            data.ColorClass = "alert-success"; // Yeşil
                            data.ActionSuggestion = "✅ Trafik temiz görünüyor. Herhangi bir müdahale gerekmez.";
                        }
                        else
                        {
                            data.ColorClass = "alert-danger"; // Kırmızı

                            // Akıllı Öneri Sistemi
                            string pred = data.Prediction.ToString();
                            if (pred.Contains("DoS"))
                                data.ActionSuggestion = "⛔ Acil: Rate-Limiting uygulayın veya kaynak IP'yi düşürün.";
                            else if (pred.Contains("Brute") || pred.Contains("Web"))
                                data.ActionSuggestion = "🔒 Kullanıcı hesabını kilitleyin ve IP adresini banlayın.";
                            else if (pred.Contains("Botnet"))
                                data.ActionSuggestion = "🤖 Cihazı ağdan izole edin (Karantina) ve C&C bağlantısını kesin.";
                            else if (pred.Contains("Scan"))
                                data.ActionSuggestion = "👁️ Keşif yapılıyor! Açık portları kontrol edin.";
                            else
                                data.ActionSuggestion = "⚠️ Şüpheli aktivite! Firewall loglarını inceleyin.";
                        }
                    }
                    else
                    {
                        throw new Exception($"Sunucu Hatası: {response.StatusCode}");
                    }
                }
                catch (Exception ex)
                {
                    // Hata durumunda program çökmesin, ekrana hatayı bassın
                    data.Prediction = "Bağlantı Hatası";
                    data.Status = "HATA";
                    data.Confidence = 0;
                    data.ActionSuggestion = $"Detay: {ex.Message}";
                    data.ColorClass = "alert-warning"; // Sarı Uyarı Rengi
                    data.IsAnalyzed = true;
                }

                // Geçmişe Ekle 
                _history.Insert(0, new LogEntry
                {
                    Time = DateTime.Now.ToString("HH:mm:ss"),
                    Status = data.Status,
                    Message = data.Status == "HATA" ? "API Bağlantı Sorunu" : $"{data.Prediction} (%{data.Confidence:0.0})",
                    ColorClass = data.Status == "GÜVENLİ" ? "text-success" : (data.Status == "HATA" ? "text-warning" : "text-danger")
                });

                // Geçmiş listesini 5 ile sınırla
                if (_history.Count > 5) _history.RemoveAt(5);
            }

            ViewBag.History = _history;
            return View("Index", data);
        }
    }
}