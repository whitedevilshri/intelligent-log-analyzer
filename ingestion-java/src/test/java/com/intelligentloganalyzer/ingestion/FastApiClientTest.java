package com.intelligentloganalyzer.ingestion;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FastApiClientTest {

    private static final String API_URL_PROPERTY = "intelligent.log.analyzer.api-url";
    private static final String API_PATH = "/api/v1/ingestion/batches";
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static HttpServer server;
    private static final AtomicInteger RESPONSE_STATUS = new AtomicInteger(202);
    private static final AtomicReference<String> FIXED_RESPONSE_BODY = new AtomicReference<>();
    private static final AtomicReference<String> LAST_METHOD = new AtomicReference<>();
    private static final AtomicReference<String> LAST_PATH = new AtomicReference<>();
    private static final AtomicReference<String> LAST_CONTENT_TYPE = new AtomicReference<>();
    private static final AtomicReference<String> LAST_ACCEPT = new AtomicReference<>();
    private static final AtomicReference<String> LAST_REQUEST_BODY = new AtomicReference<>();

    @BeforeAll
    static void startLocalServer() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext(API_PATH, FastApiClientTest::handleRequest);
        server.start();

        String endpoint = "http://127.0.0.1:" + server.getAddress().getPort() + API_PATH;
        System.setProperty(API_URL_PROPERTY, endpoint);
    }

    @AfterAll
    static void stopLocalServer() {
        if (server != null) {
            server.stop(0);
        }
        System.clearProperty(API_URL_PROPERTY);
    }

    @BeforeEach
    void resetStubResponse() {
        RESPONSE_STATUS.set(202);
        FIXED_RESPONSE_BODY.set(null);
        LAST_METHOD.set(null);
        LAST_PATH.set(null);
        LAST_CONTENT_TYPE.set(null);
        LAST_ACCEPT.set(null);
        LAST_REQUEST_BODY.set(null);
    }

    @Test
    void sendsJsonAndAcceptsValid202Response() throws Exception {
        List<Map<String, Object>> records = validRecords();

        FastApiClient.sendBatch("batch-123", records);

        assertEquals("POST", LAST_METHOD.get());
        assertEquals(API_PATH, LAST_PATH.get());
        assertTrue(LAST_CONTENT_TYPE.get().startsWith("application/json"));
        assertTrue(LAST_ACCEPT.get().contains("application/json"));

        JsonNode sentBody = MAPPER.readTree(LAST_REQUEST_BODY.get());
        assertEquals("batch-123", sentBody.path("batch_id").asText());
        assertEquals(1, sentBody.path("records").size());
    }

    @Test
    void reportsNon202HttpStatusAndResponseBody() {
        RESPONSE_STATUS.set(400);
        FIXED_RESPONSE_BODY.set("invalid record");

        IOException exception = assertThrows(
                IOException.class,
                () -> FastApiClient.sendBatch("batch-bad", validRecords())
        );

        assertTrue(exception.getMessage().contains("HTTP status: 400"));
        assertTrue(exception.getMessage().contains("invalid record"));
    }

    @Test
    void rejectsAcceptedResponseForDifferentBatchId() {
        FIXED_RESPONSE_BODY.set(
                "{\"status\":\"accepted\",\"batch_id\":\"other-batch\",\"records_received\":1}"
        );

        IOException exception = assertThrows(
                IOException.class,
                () -> FastApiClient.sendBatch("batch-expected", validRecords())
        );

        assertTrue(exception.getMessage().contains("response validation failed"));
    }

    @Test
    void rejectsAcceptedResponseWithWrongRecordCount() {
        FIXED_RESPONSE_BODY.set(
                "{\"status\":\"accepted\",\"batch_id\":\"batch-expected\",\"records_received\":99}"
        );

        IOException exception = assertThrows(
                IOException.class,
                () -> FastApiClient.sendBatch("batch-expected", validRecords())
        );

        assertTrue(exception.getMessage().contains("response validation failed"));
    }

    @Test
    void malformedJsonResponseProducesIOExceptionInsteadOfNullPointerException() {
        FIXED_RESPONSE_BODY.set("not-json");

        IOException exception = assertThrows(
                IOException.class,
                () -> FastApiClient.sendBatch("batch-malformed", validRecords())
        );

        assertTrue(exception.getMessage().contains("not valid JSON"));
    }

    @Test
    void empty202ResponseProducesIOExceptionInsteadOfNullPointerException() {
        FIXED_RESPONSE_BODY.set("");

        IOException exception = assertThrows(
                IOException.class,
                () -> FastApiClient.sendBatch("batch-empty-response", validRecords())
        );

        assertTrue(exception.getMessage().contains("response validation failed"));
    }

    @Test
    void rejectsNullOrBlankBatchId() {
        assertThrows(
                IllegalArgumentException.class,
                () -> FastApiClient.sendBatch(null, validRecords())
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> FastApiClient.sendBatch("   ", validRecords())
        );
    }

    @Test
    void rejectsNullOrEmptyRecords() {
        assertThrows(
                IllegalArgumentException.class,
                () -> FastApiClient.sendBatch("batch-empty", null)
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> FastApiClient.sendBatch("batch-empty", List.of())
        );
    }

    @Test
    void rejectsBatchesLargerThan500Records() {
        List<Map<String, Object>> records = new ArrayList<>();
        for (int index = 0; index < 501; index++) {
            records.add(Map.of("line_id", index, "content", "sample"));
        }

        assertThrows(
                IllegalArgumentException.class,
                () -> FastApiClient.sendBatch("batch-too-large", records)
        );
    }

    @Test
    void rejectsNonObjectAcceptedResponse() {
        FIXED_RESPONSE_BODY.set("[]");

        IOException exception = assertThrows(
                IOException.class,
                () -> FastApiClient.sendBatch("batch-array-response", validRecords())
        );

        assertTrue(exception.getMessage().contains("response validation failed"));
    }

    private static List<Map<String, Object>> validRecords() {
        return List.of(Map.of("line_id", 1, "content", "sample log entry"));
    }

    private static void handleRequest(HttpExchange exchange) throws IOException {
        LAST_METHOD.set(exchange.getRequestMethod());
        LAST_PATH.set(exchange.getRequestURI().getPath());
        LAST_CONTENT_TYPE.set(exchange.getRequestHeaders().getFirst("Content-Type"));
        LAST_ACCEPT.set(exchange.getRequestHeaders().getFirst("Accept"));
        LAST_REQUEST_BODY.set(new String(
                exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8
        ));

        String body = FIXED_RESPONSE_BODY.get();
        if (body == null) {
            JsonNode request = MAPPER.readTree(LAST_REQUEST_BODY.get());
            String batchId = request.path("batch_id").asText();
            int recordCount = request.path("records").isArray()
                    ? request.path("records").size()
                    : -1;
            body = MAPPER.writeValueAsString(Map.of(
                    "status", "accepted",
                    "batch_id", batchId,
                    "records_received", recordCount
            ));
        }

        byte[] responseBytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(RESPONSE_STATUS.get(), responseBytes.length);
        try (OutputStream output = exchange.getResponseBody()) {
            output.write(responseBytes);
        } finally {
            exchange.close();
        }
    }
}
