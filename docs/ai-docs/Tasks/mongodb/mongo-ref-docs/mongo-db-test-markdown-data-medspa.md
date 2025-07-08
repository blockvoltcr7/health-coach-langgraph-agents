I'll create a comprehensive test markdown document for a medspa based on the Prime IV Hydration reference. Here's a complete test script with realistic medspa content:

```python
import os
import tempfile
from datetime import datetime
from pymongo.mongo_client import MongoClient
import certifi
from pymongo.server_api import ServerApi
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create test markdown content
MEDSPA_MARKDOWN_CONTENT = """# Prime Wellness MedSpa - IV Therapy & Wellness Treatments

## About Our MedSpa

Prime Wellness MedSpa is a premier medical spa offering cutting-edge IV hydration therapy, vitamin infusions, and wellness treatments. Our board-certified medical professionals provide personalized care in a luxurious, spa-like environment.

### Our Mission

We believe in optimizing health and wellness through innovative IV therapy treatments that deliver essential nutrients directly to your bloodstream for maximum absorption and immediate benefits.

## IV Hydration Treatments

### The Executive

**Price**: $249
**Duration**: 45-60 minutes
**Key Ingredients**: B-Complex, Vitamin C, Magnesium, Calcium, B12, Glutathione Push

The Executive IV drip is designed for high-performing professionals who need sustained energy and mental clarity. This premium blend helps combat fatigue, brain fog, and stress while boosting immune function.

**Benefits**:
- Enhanced mental clarity and focus
- Increased energy levels
- Stress reduction
- Immune system support
- Improved sleep quality

**Who It's For**: Busy executives, entrepreneurs, and professionals dealing with high stress and demanding schedules.

### Beauty Glow

**Price**: $199
**Duration**: 30-45 minutes
**Key Ingredients**: Vitamin C, Biotin, Glutathione, B-Complex, Zinc

Our Beauty Glow IV infusion is formulated to enhance your natural beauty from within. This powerful blend of antioxidants and vitamins promotes healthy skin, hair, and nails.

**Benefits**:
- Brighter, more radiant skin
- Stronger hair and nails
- Reduced signs of aging
- Improved skin elasticity
- Enhanced collagen production

**Who It's For**: Anyone looking to improve their skin health, combat aging, or prepare for special events.

### Athletic Performance

**Price**: $229
**Duration**: 45 minutes
**Key Ingredients**: Amino Acids, B-Complex, Vitamin C, Magnesium, Calcium, B12, Glutathione

Designed for athletes and fitness enthusiasts, this IV therapy optimizes performance, speeds recovery, and reduces muscle fatigue.

**Benefits**:
- Faster muscle recovery
- Reduced inflammation
- Enhanced endurance
- Improved hydration
- Decreased muscle cramps

**Who It's For**: Athletes, weekend warriors, and anyone with an active lifestyle.

### Immunity Boost

**Price**: $189
**Duration**: 30-45 minutes
**Key Ingredients**: High-dose Vitamin C, Zinc, B-Complex, Vitamin D

Strengthen your immune system with our powerful immunity IV drip. Perfect for cold and flu season or when you feel run down.

**Benefits**:
- Enhanced immune function
- Reduced illness duration
- Increased antioxidant levels
- Better resistance to infections
- Improved overall wellness

**Who It's For**: Those exposed to illness, frequent travelers, or anyone wanting to boost their immune system.

### Hangover Relief

**Price**: $169
**Duration**: 30-40 minutes
**Key Ingredients**: Saline, B-Complex, Anti-nausea medication, Pain relief medication, Vitamin C

Our hangover IV provides fast relief from hangover symptoms, rehydrating your body and replenishing essential nutrients.

**Benefits**:
- Rapid rehydration
- Nausea relief
- Headache reduction
- Energy restoration
- Electrolyte balance

**Who It's For**: Anyone experiencing hangover symptoms or dehydration.

## Specialty Treatments

### NAD+ Therapy

**Price**: Starting at $599
**Duration**: 2-4 hours
**Protocol**: Series of 4-6 treatments recommended

NAD+ (Nicotinamide Adenine Dinucleotide) is a coenzyme essential for cellular energy production and DNA repair. Our NAD+ IV therapy can help reverse aging at the cellular level.

**Benefits**:
- Increased energy and mental clarity
- Improved metabolism
- Enhanced athletic performance
- Better sleep patterns
- Reduced inflammation
- Cellular regeneration

### Vitamin Injections

#### B12 Shots
**Price**: $35
**Frequency**: Weekly or bi-weekly

Boost your energy naturally with our B12 injections. Perfect for vegetarians, vegans, or anyone with B12 deficiency.

#### Glutathione Push
**Price**: $75
**Duration**: 5-10 minutes

Known as the "master antioxidant," glutathione helps detoxify the body and improve skin health.

#### Vitamin D Injection
**Price**: $45
**Frequency**: Monthly

Essential for bone health, immune function, and mood regulation.

## Wellness Programs

### Weight Management Program

Our comprehensive weight management program combines IV therapy with nutritional counseling and metabolic optimization.

**Includes**:
- Weekly Slim Shot injections (MIC + B12)
- Bi-weekly metabolic IV drips
- Nutritional consultation
- Body composition analysis
- Customized meal plans

**Price**: $899/month

### Executive Wellness Package

Designed for busy professionals who want to maintain peak performance.

**Includes**:
- 4 Executive IV drips per month
- Weekly B12 injections
- Monthly NAD+ therapy
- Concierge scheduling
- VIP lounge access

**Price**: $1,499/month

## Treatment Process

### Initial Consultation

Every new client receives a comprehensive consultation including:
- Medical history review
- Vital signs assessment
- Treatment recommendations
- Customized wellness plan

### During Your Treatment

1. **Arrival**: Check in at our reception and enjoy complimentary beverages
2. **Assessment**: Our nurse will take your vitals and review your health status
3. **IV Placement**: Using the smallest needle possible for comfort
4. **Relaxation**: Enjoy our massage chairs, WiFi, and entertainment options
5. **Monitoring**: Medical staff monitors you throughout the treatment
6. **Completion**: Gentle removal of IV and post-treatment instructions

### After Your Treatment

- Immediate effects often felt within 30 minutes
- Peak benefits typically occur 12-24 hours post-treatment
- Effects can last 5-7 days depending on the treatment
- Stay hydrated to maximize benefits

## Safety & Medical Standards

### Medical Team

Our treatments are administered by:
- Board-certified physicians
- Registered nurses
- Licensed medical professionals

### Safety Protocols

- Sterile, single-use equipment
- Pharmaceutical-grade ingredients
- Continuous vital sign monitoring
- Emergency protocols in place
- Full medical assessment before treatment

### Contraindications

IV therapy may not be suitable for individuals with:
- Kidney disease
- Heart conditions
- Certain allergies
- Pregnancy (some treatments)

Always consult with our medical team about your health conditions.

## Membership Benefits

### VIP Membership

**Price**: $199/month

**Benefits**:
- 20% off all IV treatments
- 15% off vitamin injections
- Priority booking
- Monthly B12 injection included
- Exclusive member events
- Complimentary guest pass quarterly

### Corporate Wellness

We offer corporate packages for companies wanting to invest in employee wellness:
- On-site IV therapy
- Group discounts
- Wellness seminars
- Executive packages
- Flexible scheduling

## Booking & Policies

### How to Book

- Online booking available 24/7
- Phone reservations: (555) 123-4567
- Walk-ins welcome (subject to availability)
- Mobile IV service available

### Cancellation Policy

- 24-hour cancellation notice required
- Late cancellations subject to 50% fee
- No-shows charged full treatment price

### Payment Options

- All major credit cards accepted
- HSA/FSA cards welcome
- Corporate billing available
- Package plans and memberships
- Financing options available

## FAQ

### How long do treatments take?

Most IV treatments take 30-60 minutes. NAD+ therapy requires 2-4 hours.

### How often should I get IV therapy?

Frequency depends on your individual needs. Many clients benefit from weekly or bi-weekly treatments.

### Are there side effects?

IV therapy is generally very safe. Minor side effects may include slight bruising at the injection site or a cool sensation during infusion.

### Can I drive after treatment?

Yes, you can resume normal activities immediately after most treatments.

### Is IV therapy covered by insurance?

IV therapy is typically not covered by insurance as it's considered elective wellness treatment.

## Contact Information

**Location**: 123 Wellness Boulevard, Suite 100, Beverly Hills, CA 90210
**Phone**: (555) 123-4567
**Email**: info@primewellnessmedspa.com
**Hours**: 
- Monday-Friday: 9:00 AM - 7:00 PM
- Saturday: 10:00 AM - 6:00 PM
- Sunday: 11:00 AM - 5:00 PM

**Emergency After-Hours**: (555) 123-4568
"""

class TestDataGenerator:
    def __init__(self):
        # MongoDB setup
        self.mongo_password = os.getenv("MONGO_DB_PASSWORD")
        self.mongo_uri = f"mongodb+srv://health-coach-ai-sami:{self.mongo_password}@cluster0-health-coach-a.69bhzsd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0-health-coach-ai"
        self.client = MongoClient(self.mongo_uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
        self.db = self.client["health_coach_ai"]
        self.collection = self.db["documents"]
        
        # OpenAI embeddings setup
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def create_test_markdown_file(self) -> str:
        """Create a temporary markdown file with test content"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(MEDSPA_MARKDOWN_CONTENT)
            return f.name
    
    def process_test_data(self):
        """Process the test markdown content and insert into MongoDB"""
        
        # Create temporary markdown file
        markdown_path = self.create_test_markdown_file()
        print(f"Created test markdown file: {markdown_path}")
        
        try:
            # Step 1: Split by markdown headers
            headers_to_split_on = [
                ("#", "Main Topic"),
                ("##", "Section"),
                ("###", "Subsection"),
                ("####", "Detail"),
            ]
            
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=False
            )
            
            # Split the markdown content
            with open(markdown_path, 'r') as f:
                content = f.read()
            
            header_splits = markdown_splitter.split_text(content)
            print(f"Created {len(header_splits)} header-based chunks")
            
            # Step 2: Further split by size
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                add_start_index=True
            )
            
            final_chunks = text_splitter.split_documents(header_splits)
            print(f"Created {len(final_chunks)} final chunks")
            
            # Step 3: Generate embeddings
            texts = [doc.page_content for doc in final_chunks]
            print(f"Generating embeddings for {len(texts)} chunks...")
            embeddings_list = self.embeddings.embed_documents(texts)
            
            # Step 4: Prepare documents for MongoDB
            mongo_docs = []
            for i, (doc, embedding) in enumerate(zip(final_chunks, embeddings_list)):
                # Add medspa-specific metadata
                mongo_doc = {
                    "content": doc.page_content,
                    "embedding": embedding,
                    "document_type": "medspa_services",
                    "source": "prime_wellness_medspa",
                    "metadata": {
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(final_chunks),
                        "document_title": "Prime Wellness MedSpa Services",
                        "industry": "medical_spa",
                        "services": ["iv_therapy", "vitamin_injections", "wellness_programs"],
                        "created_at": datetime.utcnow().isoformat(),
                        "embedding_model": "text-embedding-ada-002",
                        "embedding_dimensions": len(embedding)
                    }
                }
                mongo_docs.append(mongo_doc)
            
            # Step 5: Insert into MongoDB
            print(f"Inserting {len(mongo_docs)} documents into MongoDB...")
            result = self.collection.insert_many(mongo_docs)
            print(f"Successfully inserted {len(result.inserted_ids)} documents")
            
            # Clean up temp file
            os.unlink(markdown_path)
            
            return result.inserted_ids
            
        except Exception as e:
            print(f"Error processing test data: {e}")
            # Clean up temp file on error
            if os.path.exists(markdown_path):
                os.unlink(markdown_path)
            raise
    
    def test_vector_search(self):
        """Test vector search with various queries"""
        test_queries = [
            "What IV treatments help with energy and focus?",
            "How much does hangover relief cost?",
            "What are the benefits of NAD+ therapy?",
            "Tell me about beauty and skin treatments",
            "What membership options are available?",
            "How long does an IV treatment take?",
            "What vitamins are in the Athletic Performance drip?"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Perform vector search
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 50,
                        "limit": 3,
                        "filter": {"document_type": "medspa_services"}
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "content": 1,
                        "metadata": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            results = list(self.collection.aggregate(pipeline))
            
            for i, result in enumerate(results, 1):
                print(f"\nResult {i} (Score: {result['score']:.4f}):")
                print(f"Section: {result['metadata'].get('Section', 'N/A')}")
                print(f"Content Preview: {result['content'][:200]}...")

# Run the test data generator
if __name__ == "__main__":
    generator = TestDataGenerator()
    
    # Process and insert test data
    print("Processing MedSpa test data...")
    inserted_ids = generator.process_test_data()
    
    # Wait a moment for indexing
    import time
    time.sleep(2)
    
    # Test search functionality
    print("\n\nTesting vector search functionality...")
    generator.test_vector_search()
```

This test script includes:

1. **Comprehensive MedSpa Content** covering:
   - Multiple IV therapy treatments
   - Pricing and duration information
   - Detailed benefits and target audiences
   - Specialty treatments (NAD+, vitamin injections)
   - Wellness programs and packages
   - Safety protocols and medical standards
   - Membership options
   - Booking policies and FAQs

2. **Structured Markdown** with:
   - Clear hierarchy (H1-H4 headers)
   - Detailed service descriptions
   - Pricing information
   - Lists and formatting
   - Contact information

3. **Test Search Queries** to validate:
   - Service-specific searches
   - Price inquiries
   - Treatment benefits
   - General information queries

The content is designed to test the vector search's ability to:
- Find relevant treatments based on symptoms/needs
- Retrieve specific pricing information
- Match treatments to customer profiles
- Answer common questions about procedures

This will give you a realistic dataset to test MongoDB's vector search capabilities against your Supabase implementation.