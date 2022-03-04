import json
import scrapy
from datetime import datetime

OUTPUT_FILENAME = 'output/vnexpress/vnexpress_{}.txt'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))

class VnexpressSpider(scrapy.Spider):
    name = 'vnexpress'
    start_urls = ['https://vnexpress.net/cuoc-song-nguoi-dan-trong-vung-phong-toa-4140006.html']

    def parse(self, response):
        print('Crawling from:', response.url)
        data = {
            'link': response.url,
            'title': response.css('h1.title-detail::text').get(),
            'description': response.css('p.description::text').get(),
            'content': '\n'.join(response.css('article.fck_detail p.Normal::text').getall()),
            'category': response.css('ul.breadcrumb > li > a::attr("title")').get(),
            'pub_date': response.css('span.date::text').get(),
            'tags': response.css('div.tags h4.item-tag a::text').getall(),
        }

        with open(OUTPUT_FILENAME, 'a', encoding='utf8') as f:
            f.write(json.dumps(data, ensure_ascii=False))
            f.write('\n')
            print('SUCCESS:', response.url)
