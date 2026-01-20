namespace SiberGuvenlikPaneli.Models 
{
    // Formdan gelen veriler (Input)
    public class TrafficData
    {
        public string Proto { get; set; } = "TCP";
        public string Service { get; set; } = "http";
        public double Duration { get; set; } = 1.0;
        public string ConnState { get; set; } = "SF";
        public string History { get; set; } = "ShADadFf";
        public int OrigPkts { get; set; } = 10;
        public int OrigIpBytes { get; set; } = 500;
        public int RespPkts { get; set; } = 10;
        public int RespIpBytes { get; set; } = 1000;

        // API Cevabı ve Sonuç Gösterimi 
        public string? Prediction { get; set; }  // Örn: "DoS Saldırısı"
        public double Confidence { get; set; }   // Örn: 98.5
        public string? Status { get; set; }      // "GÜVENLİ" veya "TEHLİKELİ"
        public string? ActionSuggestion { get; set; } // "IP'yi engelle" vs.
        public string? ColorClass { get; set; }  // Bootstrap rengi (alert-success vs.)
        public bool IsAnalyzed { get; set; } = false; // Sonuç gösterilsin mi?
    }

    // Geçmiş Listesi
    public class LogEntry
    {
        public string Time { get; set; }
        public string Status { get; set; }
        public string Message { get; set; }
        public string ColorClass { get; set; }
    }
}