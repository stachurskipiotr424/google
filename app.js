var express = require('express');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
const {ChatOpenAI, OpenAIEmbeddings, OpenAI} = require("@langchain/openai");
const {Document} = require("@langchain/core/documents");
const {ChatPromptTemplate} = require("@langchain/core/prompts");
const {createOpenAIFunctionsAgent, AgentExecutor} = require("langchain/agents");
const {DynamicStructuredTool} = require("@langchain/core/tools");
const {z} = require('zod');
const {MemoryVectorStore} = require("langchain/vectorstores/memory");
const {loadQAStuffChain} = require("langchain/chains");

const OPENAI_API_KEY = '???';
const SEARCH_API_KEY = '???'

var app = express();
app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({extended: false}));
app.use(cookieParser());

app.get('/search', async (req, res) => {
	const search = req.query.q;
	console.log('Szukamy: ', search)

	const searchResults = await fetch(`https://serpapi.com/search.json?engine=google&location=Poland&api_key=${SEARCH_API_KEY}&q=${encodeURIComponent(search)}`)
		.then(response => response.json())
		.then(response => response.organic_results)
		.then(results => results.map(result => ({
			title: result.title,
			link: result.link,
			description: result.snippet,
			source: result.source,
			keywords: result.snippet_highlighted_words
		})));

	// console.log('Wyniki z google: ', JSON.stringify(searchResults));

	const documents = searchResults.map(result => new Document({
		pageContent: `${result.title}. ${result.description}. Link: ${result.link}`,
		metadata: {
			source: result.source,
			keywords: result?.snippet_highlighted_words || [],
			link: result.link
		}
	}));

	const store = await MemoryVectorStore.fromDocuments(documents, new OpenAIEmbeddings({
		openAIApiKey: OPENAI_API_KEY
	}));

	const relatedDocuments = await store.similaritySearch(search, 5)

	// console.log('Najlepsze wyniki z google: ', JSON.stringify(relatedDocuments));

	const qaChain = loadQAStuffChain(new OpenAI({
		openAIApiKey: OPENAI_API_KEY
	}))

	const response = await qaChain.call({
		input_documents: relatedDocuments,
		question: search
	})

	console.log('Znaleźliśmy: ', response)

	return res.send({
		data: response.text
	});
})


app.post('/', async (req, res) => {
	const question = req.body.question;
	console.log('Pytanie: ', question)

	const chat = new ChatOpenAI({
		openAIApiKey: OPENAI_API_KEY,
		modelName: 'gpt-4'
	});

	const agentMainPrompt = ChatPromptTemplate.fromMessages([
		['system', `
		Jesteś przyjaznym agentem, odpowiadaj krótko i zwięźle.
		Jeżeli ktoś Ciebie prosi o adres strony internetowej to zwróć sam adres.
		Jeżeli nie wiesz czegoś w oparciu o swoją aktualną wiedzę to użyj odpowiedniego narzędzia.
		`],
		['placeholder', '{chat_history}'],
		['human', '{input}'],
		['placeholder', '{agent_scratchpad}']
	])

	const agentTools = [
		new DynamicStructuredTool({
			name: 'wyszukaj',
			description: 'Umożliwia wyszukanie brakującej informacji w internecie',
			schema: z.object({
				pytanie: z.string().describe(`
              Informacja jaką chcemy wyszukać w internecie. 
              Informacja ta powinna być tak sparafrazowana by jej wyszukanie w wyszukiwarce internetowej będzie najłatwiejsze. 
              Powinna być zawsze w języku polskim.
              Jeżeli zapytanie dotyczy adresu URL witryny to zawsze doprecyzuj, że chodzi Ci o stronę główną tej witryny.
              `)
			}),
			func: async ({pytanie}) => {
				console.log('Pytam google: ', pytanie)

				//zakładając, że odpalamy na localhost:3000
				const {data} = await fetch(`http://localhost:3000/search?q=${encodeURIComponent(pytanie)}`).then(response => response.json())

				console.log('Dostaliśmy odpowiedź: ', data);

				return data;
			}
		})
	]

	const agent = await createOpenAIFunctionsAgent({
		llm: chat,
		tools: agentTools,
		prompt: agentMainPrompt
	});

	const agentExecutor = new AgentExecutor({
		agent: agent,
		tools: agentTools,
		verbose: false
	})

	const {output: response} = await agentExecutor.invoke({
		input: question
	})

	return res.send(response)

})

module.exports = app
