
package com.intelligentloganalyzer.ingestion;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public class App {

    private static final int BATCH_SIZE = 500;

    private static final Set<String> REQUIRED_COLUMNS = Set.of(
            "LineId", "Date", "Time", "Pid", "Level",
            "Component", "Content", "EventId", "EventTemplate"
    );

    public static void main(String[] args) throws Exception {

        if (args.length != 1) {
            throw new IllegalArgumentException(
                    "Usage: App <path-to-structured-log-csv>"
            );
        }

        Path csvPath = Path.of(args[0]).toAbsolutePath().normalize();

        if (!Files.isRegularFile(csvPath)) {
            throw new IllegalArgumentException(
                    "CSV file not found: " + csvPath
            );
        }

        long rowsRead = 0;
        long acknowledgedRecords = 0;
        int batchesSent = 0;

        Set<Long> lineIds = new HashSet<>();
        Set<String> blockIds = new HashSet<>();
        Set<String> eventIds = new HashSet<>();
        Set<String> templates = new HashSet<>();

        List<Map<String, Object>> batch =
                new ArrayList<>(BATCH_SIZE);

        try (Reader reader = Files.newBufferedReader(
                    csvPath, StandardCharsets.UTF_8);
             CSVParser parser = CSVFormat.DEFAULT.builder()
                    .setHeader()
                    .setSkipHeaderRecord(true)
                    .get()
                    .parse(reader)) {

            if (!parser.getHeaderMap().keySet()
                    .containsAll(REQUIRED_COLUMNS)) {
                throw new IllegalArgumentException(
                        "CSV is missing required columns. Found: "
                                + parser.getHeaderMap().keySet()
                );
            }

            for (CSVRecord record : parser) {

                rowsRead++;

                // Validate that every required field is present.
                for (String column : REQUIRED_COLUMNS) {
                    String value = record.get(column);

                    if (value == null || value.isBlank()) {
                        throw new IllegalArgumentException(
                                "Missing value in CSV record "
                                        + record.getRecordNumber()
                                        + ", column: " + column
                        );
                    }
                }

                long lineId;

                try {
                    lineId = Long.parseLong(record.get("LineId"));
                } catch (NumberFormatException exception) {
                    throw new IllegalArgumentException(
                            "Invalid LineId: " + record.get("LineId")
                    );
                }

                if (lineId <= 0 || !lineIds.add(lineId)) {
                    throw new IllegalArgumentException(
                            "Invalid or duplicate LineId: " + lineId
                    );
                }

                // Extract BlockIds from the log Content.
                Set<String> extractedBlockIds =
                        BlockIdExtractor.extractDistinct(
                                record.get("Content")
                        );

                if (extractedBlockIds.isEmpty()) {
                    throw new IllegalArgumentException(
                            "No BlockId found in LineId: " + lineId
                    );
                }

                if (extractedBlockIds.size() > 1) {
                    throw new IllegalArgumentException(
                            "Multiple distinct BlockIds found in LineId: "
                                    + lineId
                    );
                }

                blockIds.addAll(extractedBlockIds);
                eventIds.add(record.get("EventId"));
                templates.add(record.get("EventTemplate"));

                Map<String, Object> logRecord = Map.of(
                        "line_id", lineId,
                        "date", record.get("Date"),
                        "time", record.get("Time"),
                        "pid", record.get("Pid"),
                        "level", record.get("Level"),
                        "component", record.get("Component"),
                        "content", record.get("Content"),
                        "event_id", record.get("EventId"),
                        "event_template", record.get("EventTemplate"),
                        "block_ids", List.copyOf(extractedBlockIds)
                );

                batch.add(logRecord);

                // Send each full batch to FastAPI.
                if (batch.size() == BATCH_SIZE) {

                    batchesSent++;

                    String batchId = String.format(
                            Locale.ROOT,
                            "hdfs-100k-%04d",
                            batchesSent
                    );

                    FastApiClient.sendBatch(batchId, batch);

                    // Count records only after the server acknowledges them.
                    acknowledgedRecords += batch.size();

                    batch = new ArrayList<>(BATCH_SIZE);
                }
            }
        }

        // Send the final batch if fewer than 500 records remain.
        if (!batch.isEmpty()) {

            batchesSent++;

            String batchId = String.format(
                    Locale.ROOT,
                    "hdfs-100k-%04d",
                    batchesSent
            );

            FastApiClient.sendBatch(batchId, batch);
            acknowledgedRecords += batch.size();
        }

        System.out.println("\n=== JAVA INGESTION REPORT ===");
        System.out.println("CSV: " + csvPath);
        System.out.println("Rows read: " + rowsRead);
        System.out.println("Distinct LineIds: " + lineIds.size());
        System.out.println("Distinct BlockIds: " + blockIds.size());
        System.out.println("Distinct EventIds: " + eventIds.size());
        System.out.println("Distinct templates: " + templates.size());
        System.out.println("Batches sent: " + batchesSent);
        System.out.println(
                "Records acknowledged by FastAPI: "
                        + acknowledgedRecords
        );

        if (rowsRead != acknowledgedRecords) {
            throw new IllegalStateException(
                    "Record count mismatch!"
            );
        }

        System.out.println("Validation: PASS");
    }
}
