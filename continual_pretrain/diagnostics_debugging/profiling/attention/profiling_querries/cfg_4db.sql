SELECT
    s.value                                    AS kernel,
    ROUND((k.end - k.start) / 1000.0, 2)      AS gpu_dur_us,
    k.gridX, k.gridY, k.gridZ,
    m.value                                    As mangledName

FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId = r.correlationId
JOIN StringIds s ON s.id = k.shortName
JOIN StringIds m ON m.id = k.mangledName

WHERE r.start >= (SELECT start FROM NVTX_EVENTS
                  WHERE text LIKE 'attn_scores%'
                  AND start > 50000000000 LIMIT 1 OFFSET 50)
  AND r.end   <= (SELECT end   FROM NVTX_EVENTS
                  WHERE text LIKE 'attn_scores%'
                  AND start > 50000000000 LIMIT 1 OFFSET 50)
ORDER BY k.start;
