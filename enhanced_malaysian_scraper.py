"""
Enhanced Malaysian Government Scraper
Updated to better handle AGC website structure and find more legal documents
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMalaysianLegalScraper:
    """
    Enhanced scraper for Malaysian legal documents
    Targets multiple sources and improved parsing
    """
    
    def __init__(self, download_dir="./malaysian_legal_pdfs"):
        self.base_url = "https://lom.agc.gov.my"
        
        # Multiple target URLs for comprehensive scraping
        self.target_urls = [
            "https://lom.agc.gov.my/principal.php?type=updated",  # Principal Acts
            "https://lom.agc.gov.my/principal.php",               # All Principal Acts
            "https://lom.agc.gov.my/constitution.php",            # Constitution
        ]
        
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True, parents=True)
        
        # Create organized subdirectories
        self.en_dir = self.download_dir / "english"
        self.bm_dir = self.download_dir / "bahasa_malaysia"
        self.constitution_dir = self.download_dir / "constitution"
        
        for directory in [self.en_dir, self.bm_dir, self.constitution_dir]:
            directory.mkdir(exist_ok=True)
        
        # Enhanced session with better headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,ms;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
        })
        
        # Track downloads
        self.downloaded_pdfs = []
        self.failed_downloads = []
        self.processed_urls = set()
    
    def discover_all_legal_pages(self):
        """Discover all pages containing legal documents"""
        logger.info("🔍 Discovering all Malaysian legal document pages...")
        
        all_pages = []
        
        for target_url in self.target_urls:
            logger.info(f"📄 Scanning: {target_url}")
            
            try:
                response = self.session.get(target_url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Different strategies for different pages
                if 'constitution' in target_url:
                    pages = self.parse_constitution_page(soup, target_url)
                else:
                    pages = self.parse_acts_page(soup, target_url)
                
                all_pages.extend(pages)
                logger.info(f"✅ Found {len(pages)} pages from {target_url}")
                
                time.sleep(1)  # Be respectful
                
            except Exception as e:
                logger.error(f"❌ Error accessing {target_url}: {e}")
        
        logger.info(f"📚 Total pages discovered: {len(all_pages)}")
        return all_pages
    
    def parse_constitution_page(self, soup, url):
        """Parse constitution-specific pages"""
        pages = [{
            'url': url,
            'soup': soup,
            'type': 'constitution',
            'title': 'Federal Constitution'
        }]
        return pages
    
    def parse_acts_page(self, soup, url):
        """Parse acts listing pages with enhanced detection"""
        pages = [{
            'url': url,
            'soup': soup,
            'type': 'acts',
            'title': 'Principal Acts'
        }]
        
        # Look for pagination or additional act pages
        # Check for links to specific acts
        act_detail_links = soup.find_all('a', href=lambda x: x and (
            'view.php' in x or 
            'act_view' in x or 
            'details' in x or
            re.search(r'act.*\d+', x, re.I)
        ))
        
        for link in act_detail_links[:10]:  # Limit to prevent too many requests
            href = link.get('href')
            if href and href not in self.processed_urls:
                full_url = urljoin(url, href)
                self.processed_urls.add(href)
                
                try:
                    response = self.session.get(full_url, timeout=20)
                    response.raise_for_status()
                    
                    act_soup = BeautifulSoup(response.content, 'html.parser')
                    pages.append({
                        'url': full_url,
                        'soup': act_soup,
                        'type': 'act_detail',
                        'title': link.get_text(strip=True)
                    })
                    
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Could not access act detail page {full_url}: {e}")
        
        return pages
    
    def extract_all_pdf_links(self, pages):
        """Enhanced PDF link extraction with multiple strategies"""
        logger.info("🔗 Extracting PDF links with enhanced detection...")
        
        pdf_links = []
        
        for page_info in pages:
            soup = page_info['soup']
            page_url = page_info['url']
            page_type = page_info.get('type', 'unknown')
            
            logger.info(f"🔍 Processing {page_type} page: {page_url}")
            
            # Strategy 1: Direct PDF links
            direct_pdf_links = soup.find_all('a', href=lambda x: x and x.lower().endswith('.pdf'))
            
            # Strategy 2: Links with PDF-related text
            pdf_text_links = soup.find_all('a', text=re.compile(r'(pdf|download|reprint)', re.I))
            
            # Strategy 3: Links with download-related attributes
            download_links = soup.find_all('a', href=lambda x: x and (
                'download' in x.lower() or 
                'pdf' in x.lower() or
                'file' in x.lower()
            ))
            
            # Strategy 4: Form submissions or JavaScript links (convert to direct links)
            form_links = soup.find_all(['input', 'button'], type='submit')
            
            # Combine all found links
            all_found_links = direct_pdf_links + pdf_text_links + download_links
            
            for link in all_found_links:
                href = link.get('href', '')
                text = link.get_text(strip=True)
                
                if href:
                    full_url = urljoin(page_url, href)
                    
                    # Enhanced act info parsing
                    act_info = self.enhanced_parse_act_info(text, href, page_info)
                    
                    pdf_links.append({
                        'url': full_url,
                        'text': text,
                        'source_page': page_url,
                        'page_type': page_type,
                        'act_number': act_info.get('act_number'),
                        'act_name': act_info.get('act_name'),
                        'language': act_info.get('language', 'unknown'),
                        'document_type': act_info.get('document_type', 'act'),
                        'is_direct_pdf': href.lower().endswith('.pdf')
                    })
        
        # Remove duplicates
        unique_pdf_links = []
        seen_urls = set()
        
        for link in pdf_links:
            if link['url'] not in seen_urls:
                unique_pdf_links.append(link)
                seen_urls.add(link['url'])
        
        logger.info(f"🔗 Found {len(unique_pdf_links)} unique PDF links")
        return unique_pdf_links
    
    def enhanced_parse_act_info(self, link_text, href, page_info):
        """Enhanced parsing of act information"""
        act_info = {
            'act_number': None,
            'act_name': None,
            'language': 'unknown',
            'document_type': 'act'
        }
        
        text_lower = link_text.lower()
        href_lower = href.lower()
        page_type = page_info.get('type', '')
        
        # Detect document type
        if page_type == 'constitution' or 'constitution' in text_lower:
            act_info['document_type'] = 'constitution'
        elif 'regulation' in text_lower or 'rules' in text_lower:
            act_info['document_type'] = 'regulation'
        else:
            act_info['document_type'] = 'act'
        
        # Enhanced language detection
        if any(word in text_lower for word in ['english', 'en', '(en)', 'eng']):
            act_info['language'] = 'english'
        elif any(word in text_lower for word in ['bahasa', 'bm', '(bm)', 'malay', 'malaysia']):
            act_info['language'] = 'bahasa_malaysia'
        elif any(word in href_lower for word in ['_en.', '_eng.', 'english']):
            act_info['language'] = 'english'
        elif any(word in href_lower for word in ['_bm.', '_my.', 'bahasa']):
            act_info['language'] = 'bahasa_malaysia'
        
        # Enhanced Act number extraction
        # Look for patterns like "Act 265", "265/1955", "No. 265"
        act_patterns = [
            r'act\s*(\d+)',
            r'akta\s*(\d+)',
            r'(\d+)/\d{4}',  # 265/1955 format
            r'no\.?\s*(\d+)',
            r'number\s*(\d+)'
        ]
        
        for pattern in act_patterns:
            match = re.search(pattern, text_lower)
            if match:
                act_info['act_number'] = match.group(1)
                break
        
        # Clean act name
        act_name = link_text
        # Remove language indicators
        act_name = re.sub(r'\s*\(?(english|bahasa|en|bm|download|pdf|reprint)\)?\s*', '', act_name, flags=re.I)
        # Remove extra whitespace
        act_name = re.sub(r'\s+', ' ', act_name).strip()
        
        if act_name and act_name != link_text:
            act_info['act_name'] = act_name
        else:
            act_info['act_name'] = page_info.get('title', 'Unknown Act')
        
        return act_info
    
    def download_pdf_enhanced(self, pdf_info):
        """Enhanced PDF download with better error handling"""
        url = pdf_info['url']
        act_number = pdf_info.get('act_number', 'Unknown')
        act_name = pdf_info.get('act_name', 'Unknown_Document')
        language = pdf_info.get('language', 'unknown')
        doc_type = pdf_info.get('document_type', 'act')
        
        # Create clean filename
        safe_name = re.sub(r'[^\w\s-]', '', act_name)
        safe_name = re.sub(r'[-\s]+', '_', safe_name)
        safe_name = safe_name[:100]  # Limit length
        
        # Choose directory based on document type and language
        if doc_type == 'constitution':
            filename = f"Constitution_{safe_name}_{language.upper()}.pdf"
            save_dir = self.constitution_dir
        elif language == 'english':
            filename = f"Act_{act_number}_{safe_name}_EN.pdf"
            save_dir = self.en_dir
        elif language == 'bahasa_malaysia':
            filename = f"Act_{act_number}_{safe_name}_BM.pdf"
            save_dir = self.bm_dir
        else:
            filename = f"{doc_type.title()}_{act_number}_{safe_name}_Unknown.pdf"
            save_dir = self.download_dir
        
        filepath = save_dir / filename
        
        # Skip if already downloaded
        if filepath.exists():
            logger.info(f"⏭️ Already exists: {filename}")
            return True
        
        try:
            logger.info(f"⬇️ Downloading: {filename}")
            
            # Handle indirect links with multiple fallback strategies
            final_url = url
            
            if not pdf_info['is_direct_pdf']:
                final_url = self.resolve_pdf_url(url)
                if not final_url:
                    logger.warning(f"❌ Could not resolve PDF URL: {url}")
                    return False
            
            # Download with retry logic
            for attempt in range(3):
                try:
                    response = self.session.get(final_url, timeout=60, stream=True)
                    response.raise_for_status()
                    
                    # Verify it's a PDF
                    content_type = response.headers.get('content-type', '').lower()
                    if 'pdf' not in content_type and not final_url.lower().endswith('.pdf'):
                        # Check first few bytes for PDF signature
                        first_chunk = next(response.iter_content(chunk_size=1024), b'')
                        if not first_chunk.startswith(b'%PDF'):
                            logger.warning(f"❌ Not a PDF file: {final_url}")
                            return False
                        
                        # Reset response for full download
                        response = self.session.get(final_url, timeout=60, stream=True)
                        response.raise_for_status()
                    
                    # Save the file
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    file_size = filepath.stat().st_size / 1024  # KB
                    
                    # Verify download
                    if file_size < 1:  # Less than 1KB is probably not a real PDF
                        filepath.unlink()
                        logger.warning(f"❌ Downloaded file too small: {filename}")
                        return False
                    
                    logger.info(f"✅ Downloaded: {filename} ({file_size:.1f} KB)")
                    
                    self.downloaded_pdfs.append({
                        'filename': filename,
                        'filepath': str(filepath),
                        'url': final_url,
                        'original_url': url,
                        'size_kb': file_size,
                        'act_number': act_number,
                        'act_name': act_name,
                        'language': language,
                        'document_type': doc_type,
                        'download_time': datetime.now().isoformat()
                    })
                    
                    return True
                    
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"⚠️ Download attempt {attempt + 1} failed, retrying: {e}")
                        time.sleep(5)
                    else:
                        raise e
            
        except Exception as e:
            logger.error(f"❌ Failed to download {filename}: {e}")
            self.failed_downloads.append({
                'url': url,
                'filename': filename,
                'error': str(e)
            })
            return False
    
    def resolve_pdf_url(self, page_url):
        """Try to find the actual PDF URL from a page"""
        try:
            response = self.session.get(page_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for direct PDF links
            pdf_link = soup.find('a', href=lambda x: x and x.lower().endswith('.pdf'))
            if pdf_link:
                return urljoin(page_url, pdf_link['href'])
            
            # Look for download buttons/forms
            download_forms = soup.find_all('form', action=lambda x: x and ('download' in x.lower() or 'pdf' in x.lower()))
            if download_forms:
                # Try to construct download URL
                form = download_forms[0]
                action = form.get('action', '')
                if action:
                    return urljoin(page_url, action)
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Could not resolve PDF URL for {page_url}: {e}")
            return None
    
    def scrape_all_enhanced(self):
        """Enhanced main scraping method"""
        logger.info("🇲🇾 Starting Enhanced Malaysian Legal Documents Scraper")
        logger.info(f"📁 Download directory: {self.download_dir}")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Discover all legal document pages
            pages = self.discover_all_legal_pages()
            if not pages:
                logger.error("❌ No pages found - check website structure")
                return False
            
            # Step 2: Extract all PDF links
            pdf_links = self.extract_all_pdf_links(pages)
            if not pdf_links:
                logger.error("❌ No PDF links found - check parsing logic")
                return False
            
            # Step 3: Download all PDFs
            logger.info(f"📥 Starting download of {len(pdf_links)} PDFs...")
            
            for i, pdf_info in enumerate(pdf_links, 1):
                logger.info(f"📄 [{i}/{len(pdf_links)}] Processing: {pdf_info['text'][:50]}...")
                
                self.download_pdf_enhanced(pdf_info)
                
                # Be respectful - delay between downloads
                time.sleep(3)
                
                # Progress update
                if i % 5 == 0:
                    logger.info(f"📊 Progress: {i}/{len(pdf_links)} processed, {len(self.downloaded_pdfs)} successful")
            
            # Step 4: Generate summary
            self.generate_enhanced_summary(start_time)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced scraping failed: {e}")
            return False
    
    def generate_enhanced_summary(self, start_time):
        """Generate enhanced summary report"""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            'scrape_date': end_time.isoformat(),
            'duration_seconds': duration,
            'total_downloaded': len(self.downloaded_pdfs),
            'total_failed': len(self.failed_downloads),
            'english_pdfs': len([p for p in self.downloaded_pdfs if p['language'] == 'english']),
            'bahasa_pdfs': len([p for p in self.downloaded_pdfs if p['language'] == 'bahasa_malaysia']),
            'constitution_pdfs': len([p for p in self.downloaded_pdfs if p['document_type'] == 'constitution']),
            'acts_pdfs': len([p for p in self.downloaded_pdfs if p['document_type'] == 'act']),
            'total_size_mb': sum(p['size_kb'] for p in self.downloaded_pdfs) / 1024,
            'downloaded_files': self.downloaded_pdfs,
            'failed_downloads': self.failed_downloads
        }
        
        # Save detailed summary
        summary_file = self.download_dir / "enhanced_scraping_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print summary
        logger.info("📊 ENHANCED SCRAPING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"⏱️ Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"✅ Successfully downloaded: {summary['total_downloaded']} PDFs")
        logger.info(f"❌ Failed downloads: {summary['total_failed']} PDFs")
        logger.info(f"🇬🇧 English PDFs: {summary['english_pdfs']}")
        logger.info(f"🇲🇾 Bahasa Malaysia PDFs: {summary['bahasa_pdfs']}")
        logger.info(f"📜 Constitution PDFs: {summary['constitution_pdfs']}")
        logger.info(f"⚖️ Acts PDFs: {summary['acts_pdfs']}")
        logger.info(f"💾 Total size: {summary['total_size_mb']:.1f} MB")
        logger.info(f"📄 Summary saved: {summary_file}")
        
        # Show document breakdown
        if self.downloaded_pdfs:
            logger.info("\n📚 Downloaded documents by type:")
            by_type = {}
            for pdf in self.downloaded_pdfs:
                doc_type = f"{pdf['document_type']} ({pdf['language']})"
                by_type[doc_type] = by_type.get(doc_type, 0) + 1
            
            for doc_type, count in by_type.items():
                logger.info(f"   • {doc_type}: {count} documents")


def main():
    """Run the enhanced Malaysian legal documents scraper"""
    print("🇲🇾 ENHANCED MALAYSIAN GOVERNMENT LEGAL DOCUMENTS SCRAPER")
    print("=" * 70)
    print("Target: Attorney General's Chambers (AGC) - Multiple Sources")
    print("Goal: Comprehensive download of Malaysian law PDFs for Legal AI")
    print()
    
    # Initialize enhanced scraper
    scraper = EnhancedMalaysianLegalScraper()
    
    # Start enhanced scraping
    success = scraper.scrape_all_enhanced()
    
    if success:
        print(f"\n🎉 ENHANCED SCRAPING COMPLETED!")
        print(f"📁 PDFs saved to: {scraper.download_dir}")
        print(f"📊 Downloaded: {len(scraper.downloaded_pdfs)} documents")
        print("\n📋 Next steps:")
        print("1. Review downloaded PDFs in organized folders")
        print("2. Run enhanced PDF parser to extract legal text")
        print("3. Process with your Malaysian Legal AI system")
        print("4. Replace templates in malaysian_legal_knowledge/ with real docs")
    else:
        print("\n❌ ENHANCED SCRAPING FAILED!")
        print("Check logs for errors and retry")

if __name__ == "__main__":
    main()
