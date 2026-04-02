# Projeye Katkıda Bulunma
Öncelikle bu projeye zaman ayırdığınız ve katkıda bulunmak istediğiniz için teşekkür ederiz!

Bu belge, projeye sorunsuz bir şekilde katkıda bulunabilmeniz için izlemeniz gereken adımları ve proje standartlarını içermektedir.

## Geliştirme Süreci
Bu projenin temel odak noktası, dış kütüphanelere ve hazır paketlere olan bağımlılığı en aza indirerek yapay zeka çekirdek mimarilerini sıfırdan inşa etmektir.
* Yeni bir özellik eklerken ağır dış bağımlılıklar getirmekten kaçının.
* Yorum satırlarını yanlızca gerekli yerlerde kullanmaya özen gösterin.
* Yeni modüllerin test edilmiş ve sorunsuz çalışır olduğundan emin olun.

## Hata Bildirimi ve Çözülmesi
Karşılaştığınız bir hatayı düzeltmek veya bildirmek için şu adımları izleyin:
1. **Hata Kontrolü:** Karşılaştığınız sorunun ile ilgili daha önce Issue açılıp açılmadığını kontrol edin.
2. **Issue Açma:** Hata daha önce bildirilmediyse yeni bir Issue açın. Hata ile ilgili beklenen sonucu ve karşılaştığınız sonucu ve hatayı gidermek için izleyebileceğimiz adımları detaylıca yazın.
3. **Çözüm Önerisi:** Hata ile ilgili çözüm önerilerinizi veye hatayı üzerinde kendiniz çalışıyorsanız ilgili Issue altına yorum olarak bildirin. Ardından aşağıdaki katkı sürecini izleyerek bir Pull Request açın.

## Yeni Özellik Ekleme ve Kod Katkısı Süreci
Yeni bir özellik eklemek veya kodda bir iyileştirme yapmak için doğrudan ana depoya kod gönderemezsiniz. Lütfen aşağıdaki **Fork > Test > Pull Request** iş akışını takip edin:

1. Projeyi Forklayın
Ana projeyi kendi GitHub hesabınıza Forklayın

2. Klonlayın ve Dal Oluşturun
Forkladığınız projeyi çalışma ortamınızda klonlayın ve ardından üzerinde çalışacağınız yeni bir dal oluşturun. Main ve Master dalları üstünde çalışmaktan kaçının.

3. Geliştirme ve Test
* Gerekli geliştirme ve eklemeleri yapın.
* Yaptığınız geliştirme ve eklemeleri mevcut depoyu bozmadığından emin olmak için yerel çalışma ortamınızda **mutlaka test edin.**

4. Değişiklikleri Gönderin
Yaptığınız değişiklikleri anlaşılır commit mesajlarıyla kaydedin ve kendi forkladığınız deponuza gönderin.

5. Pull Request Açın
* Orjinal projenin GitHub sayfasına gidin ve kendi deponuzdaki dalı, orijinal projenin ana dalı ile karşılaştırarak Pull Request oluşturun.
* Pull Request açıklamasında nelerin değiştirdiğinizi, hangi problemi çözdüğünüzü (eğer bir Issue çözüyorsa Fixes #IssueNumarası şeklinde) ve yerel ortamınızda hangi testleri uyguladığınızı detaylıca yazın.

## Pull Request Değerlendirme Standartları
Açılan PR'lar proje yöneticileri tarafından incelenir. Pull Requestinizin şu şartları sağlaması beklenir:
* **Test Edilmiş Olmalı:** Çalışmama veya mevcut mimariyi bozma potansiyeli içeren değişiklikler kabul edilmez.
* **Temiz Kod:** Geliştirme standartlarına ve dosya yapısına uygun olmalıdır.
* **Açıklayıcı Olmalı:** Ne yapıldığı PR metninde net bir şekilde ifade edilmelidir.

İnceleme sonucunda sizden Review Changes olarak bazı düzeltmeler veya değişiklikler istenebilir.

---

Katkılarınız için **Teşekkürler**
